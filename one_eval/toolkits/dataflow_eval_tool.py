from __future__ import annotations

import os
import time
import traceback
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import re

import pandas as pd
from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, APILLMServing_request
from dataflow.core import LLMServingABC

from one_eval.core.state import BenchInfo, ModelConfig
from one_eval.logger import get_logger

log = get_logger("DataFlowEvalTool")


class _APILLMServingWithTimeout(APILLMServing_request):
    """APILLMServing_request 的子类：让 timeout 可由实例配置。

    上游库的 timeout 是硬编码（api_chat=60s、_api_chat_with_id=1800s），
    本子类重写两个方法，使用 self._request_timeout。
    """

    def __init__(self, *args, request_timeout: float = 30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_timeout = float(request_timeout) if request_timeout and request_timeout > 0 else 30.0

    def api_chat(self, system_info: str, messages: str, model: str):
        import requests as _requests, json as _json, logging as _logging
        try:
            payload = _json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": messages},
                ],
                "temperature": 0.0,
            })
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            }
            response = _requests.post(self.api_url, headers=headers, data=payload, timeout=self._request_timeout)
            if response.status_code == 200:
                return self.format_response(response.json())
            _logging.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
        except Exception as e:
            _logging.error(f"API request error: {e}")
            return None

    def _api_chat_with_id(self, id, payload, model, is_embedding: bool = False, json_schema: dict = None):
        import requests as _requests, json as _json, logging as _logging
        try:
            if is_embedding:
                payload = _json.dumps({"model": model, "input": payload})
            elif json_schema is None:
                payload = _json.dumps({"model": model, "messages": payload})
            else:
                payload = _json.dumps({
                    "model": model,
                    "messages": payload,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "custom_response", "strict": True, "schema": json_schema},
                    },
                })
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            }
            response = _requests.post(self.api_url, headers=headers, data=payload, timeout=self._request_timeout)
            if response.status_code == 200:
                return id, self.format_response(response.json(), is_embedding)
            _logging.error(f"API request failed with status {response.status_code}: {response.text}")
            return id, None
        except Exception as e:
            _logging.error(f"API request error: {e}")
            return id, None


class DataFlowEvalTool:
    """
    封装 DataFlow 的评测 Pipeline
    - BenchAnswerGenerator
    - UnifiedBenchDatasetEvaluator
    - 支持多模型并行评测
    """

    # Class-level cache for multiple LLM servings (key = config hash)
    _cached_llm_servings: Dict[str, LLMServingABC] = {}
    # 保护 _cached_llm_servings 和 os.environ["DF_API_KEY"] 的并发写入
    _init_lock = threading.Lock()

    @classmethod
    def _make_config_key(cls, config: ModelConfig) -> str:
        """生成模型配置的唯一缓存键"""
        key_parts = [
            config.model_name_or_path,
            str(config.is_api),
            config.api_url or "",
            str(config.tensor_parallel_size),
        ]
        return "|".join(key_parts)

    def __init__(self, output_root: str = "cache/eval_results"):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def _get_or_init_llm_serving(self, config: ModelConfig) -> LLMServingABC:
        """获取或初始化 LLM Serving（线程安全，返回实例而非写入 self）

        多模型并行评测时，多个线程同时调用 run_eval，
        如果直接写 self.llm_serving 会导致线程间互相覆盖。
        因此改为返回局部 serving 实例，调用方自行持有引用。
        """
        config_key = self._make_config_key(config)

        # 快速路径：无锁检查缓存
        if config_key in DataFlowEvalTool._cached_llm_servings:
            cached = DataFlowEvalTool._cached_llm_servings[config_key]
            # 检查是否损坏（仅本地模型）
            if isinstance(cached, LocalModelLLMServing_vllm):
                if getattr(cached, "backend_initialized", False) and not hasattr(cached, "tokenizer"):
                    log.warning(f"Detected broken cached serving for {config_key}, rebuilding...")
                    try:
                        if hasattr(cached, "cleanup"):
                            cached.cleanup()
                    except Exception:
                        pass
                    del DataFlowEvalTool._cached_llm_servings[config_key]
                else:
                    log.info(f"Using cached serving for {config_key}")
                    return cached
            else:
                log.info(f"Using cached serving for {config_key}")
                return cached

        model_name_or_path = config.model_name_or_path
        if isinstance(model_name_or_path, str) and model_name_or_path:
            p = model_name_or_path.strip()
            if os.name == "nt":
                m = re.match(r"^/mnt/([a-zA-Z])/(.+)$", p)
                if m:
                    drive = m.group(1).upper()
                    rest = m.group(2).replace("/", "\\")
                    p = f"{drive}:\\{rest}"
            else:
                m = re.match(r"^([a-zA-Z]):\\(.+)$", p)
                if m:
                    drive = m.group(1).lower()
                    rest = m.group(2).replace("\\", "/")
                    p = f"/mnt/{drive}/{rest}"
            model_name_or_path = p

        log.info(f"Initializing LLM Serving: {model_name_or_path} (is_api={config.is_api})")

        serving = None
        if config.is_api:
            # APILLMServing_request reads key from env var DF_API_KEY, not from params
            # 加锁保护 os.environ 写入 + __init__ 读取，防止并发覆盖
            api_url = config.api_url or ""
            if api_url and not api_url.endswith("/chat/completions"):
                api_url = api_url.rstrip("/") + "/chat/completions"
                log.info(f"Normalized api_url to: {api_url}")
            with DataFlowEvalTool._init_lock:
                if config.api_key:
                    os.environ["DF_API_KEY"] = config.api_key
                serving = _APILLMServingWithTimeout(
                    api_url=api_url,
                    model_name=model_name_or_path,
                    max_workers=getattr(config, "api_concurrency", 1) or 1,
                    max_retries=getattr(config, "api_max_retries", 3) or 3,
                    request_timeout=getattr(config, "api_timeout", 30.0) or 30.0,
                )
        else:
            serving = LocalModelLLMServing_vllm(
                hf_model_name_or_path=model_name_or_path,
                vllm_tensor_parallel_size=config.tensor_parallel_size,
                vllm_max_tokens=config.max_tokens,
                vllm_temperature=config.temperature,
                vllm_top_p=config.top_p,
                vllm_top_k=getattr(config, "top_k", -1),
                vllm_repetition_penalty=getattr(config, "repetition_penalty", 1.0),
                vllm_seed=getattr(config, "seed", None),
                vllm_max_model_len=getattr(config, "max_model_len", None),
                vllm_gpu_memory_utilization=getattr(config, "gpu_memory_utilization", 0.9),
            )
            try:
                serving.start_serving()
                if not hasattr(serving, "tokenizer"):
                    raise RuntimeError("vLLM serving initialized without tokenizer")
            except Exception as e:
                try:
                    if hasattr(serving, "backend_initialized"):
                        serving.backend_initialized = False
                except Exception:
                    pass
                # 清理缓存
                if config_key in DataFlowEvalTool._cached_llm_servings:
                    del DataFlowEvalTool._cached_llm_servings[config_key]
                raise RuntimeError(f"Local vLLM serving init failed: {e}") from e

        # Update class-level cache (multi-model)
        with DataFlowEvalTool._init_lock:
            DataFlowEvalTool._cached_llm_servings[config_key] = serving
        log.info(f"Cached serving for {config_key}")
        return serving

    def _preprocess_dataframe(self, df, bench_name, key_mapping, cache_path="", eval_type=""):
        """Ad-hoc 数据预处理"""

        # 0. 修正 input_question_key：确保 key_mapping 中的 question key
        # 与 dataframe 中实际的列名一致
        input_question_key = key_mapping.get("input_question_key")
        if input_question_key and input_question_key not in df.columns:
            # key_mapping 记录的列名在 df 中不存在，尝试用 "question" 列
            if "question" in df.columns:
                key_mapping["input_question_key"] = "question"
                log.info(f"[{bench_name}] Corrected input_question_key from '{input_question_key}' to 'question'")

        # 1. 自动合并 choices
        choices_key = key_mapping.get("input_choices_key")
        if isinstance(choices_key, list):
            # 检查这些列是否都在 df 中
            missing_cols = [c for c in choices_key if c not in df.columns]
            if not missing_cols:
                # 合并列
                df["merged_choices"] = df.apply(lambda row: [str(row[c]) for c in choices_key], axis=1)
                key_mapping["input_choices_key"] = "merged_choices"
                log.info(f"[{bench_name}] Auto-merged columns {choices_key} into 'merged_choices'")
            else:
                log.warning(f"[{bench_name}] Cannot merge choices, missing columns: {missing_cols}")

        # 2. 自动注入 choices (针对 key3_q_choices_a)
        if eval_type == "key3_q_choices_a":
            # 如果 input_choices_key 缺失，或者对应的列不存在
            current_choices_key = key_mapping.get("input_choices_key")
            if not current_choices_key or (isinstance(current_choices_key, str) and current_choices_key not in df.columns):
                # 尝试推断是否为 Bool/Binary 任务
                # 简单启发式：检查 label 列是否存在，且值域是否类似 0/1 或 False/True
                # 为了安全，我们只对明确缺失 choices 的情况注入 ["False", "True"]
                # 这是一个合理的默认值，即便对于 Yes/No 任务，通常也是映射到 False/True 的
                if "choices" not in df.columns:
                    df["choices"] = [["False", "True"]] * len(df)
                    key_mapping["input_choices_key"] = "choices"
                    log.info(f"[{bench_name}] Auto-injected default choices ['False', 'True'] for key3_q_choices_a")

        return df, key_mapping

    def _extract_path_value(self, obj: Any, path: str) -> Any:
        if not path or not isinstance(path, str):
            return None
        cur = obj
        for p in path.split("."):
            if isinstance(cur, dict):
                if p not in cur:
                    return None
                cur = cur[p]
                continue
            if isinstance(cur, list):
                if not p.isdigit():
                    return None
                idx = int(p)
                if idx < 0 or idx >= len(cur):
                    return None
                cur = cur[idx]
                continue
            return None
        return cur

    def _materialize_nested_keys(self, source_path: str, key_paths: List[str], target_path: str) -> str:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(source_path, "r", encoding="utf-8") as rf, open(target_path, "w", encoding="utf-8") as wf:
            for line in rf:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                if isinstance(row, dict):
                    for kp in key_paths:
                        if kp and "." in kp and kp not in row:
                            row[kp] = self._extract_path_value(row, kp)
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target_path

    def _count_jsonl_rows(self, path: str) -> int:
        if not path or not os.path.exists(path):
            return 0
        cnt = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    cnt += 1
        return cnt

    def run_eval(self, bench: BenchInfo, model_config: ModelConfig, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        执行单个 Bench 的评测
        Returns:
            {
                "stats": dict,  # 评测统计结果
                "detail_path": str,  # step2 结果文件路径
                "key_mapping": dict  # 最终使用的 key_mapping
            }
        """
        if not bench.dataset_cache or not os.path.exists(bench.dataset_cache):
            raise FileNotFoundError(f"Bench {bench.bench_name} data not found at {bench.dataset_cache}")

        if not bench.bench_dataflow_eval_type:
            raise ValueError(f"Bench {bench.bench_name} missing bench_dataflow_eval_type")

        # 1. 准备 Serving（用局部变量持有，避免多线程并发时 self.llm_serving 被覆盖）
        llm_serving = self._get_or_init_llm_serving(model_config)

        # 2. 准备路径（包含模型名）
        timestamp = int(time.time())
        safe_name = bench.bench_name.replace("/", "__")
        model_safe_name = model_config.model_name_or_path.replace("/", "__").replace(":", "_")[:50]  # 截断防止过长

        # 中间结果目录
        step_cache_dir = os.path.join(self.output_root, f"{safe_name}_{model_safe_name}_{timestamp}_steps")
        os.makedirs(step_cache_dir, exist_ok=True)

        # 最终结果文件
        eval_result_path = os.path.join(self.output_root, f"{safe_name}_{model_safe_name}_{timestamp}_result.jsonl")
        nested_stage_path = os.path.join(step_cache_dir, "step_input_nested.jsonl")

        def _emit(stage: str, generated: int = 0, total: int = 0, percent: float = 0.0):
            if progress_callback:
                progress_callback({
                    "bench_name": bench.bench_name,
                    "stage": stage,
                    "generated": int(generated),
                    "total": int(total),
                    "percent": float(percent),
                })

        # 3. 准备参数映射
        key_mapping = bench.meta.get("key_mapping", {})
        log.info(f"[{bench.bench_name}] Initial Key Mapping: {key_mapping}")

        all_key_paths = [v for v in key_mapping.values() if isinstance(v, str) and v.strip()]
        nested_paths = [p for p in all_key_paths if "." in p]
        input_dataset_path = bench.dataset_cache
        if nested_paths:
            try:
                input_dataset_path = self._materialize_nested_keys(bench.dataset_cache, nested_paths, nested_stage_path)
                log.info(f"[{bench.bench_name}] Materialized nested keys: {nested_paths}")
            except Exception as e:
                log.warning(f"[{bench.bench_name}] Materialize nested keys failed, fallback to raw dataset: {e}")
                input_dataset_path = bench.dataset_cache

        # 4. 初始化 Storage
        # cache_type="jsonl" 对应 .jsonl 文件

        # === Ad-hoc 预处理 ===
        # 读取初始数据，进行必要的列注入，写到一个独立的预处理文件
        # 注意：不能使用 storage.write()，因为 operator_step=-1 时会覆盖原始文件
        preprocessed_path = input_dataset_path
        try:
            df = pd.read_json(input_dataset_path, lines=True)
            df, key_mapping = self._preprocess_dataframe(
                df,
                bench.bench_name,
                key_mapping,
                cache_path=input_dataset_path,
                eval_type=bench.bench_dataflow_eval_type
            )
            # 写入独立预处理文件，避免覆盖原始数据
            preprocessed_path = os.path.join(step_cache_dir, "step_step0.jsonl")
            os.makedirs(step_cache_dir, exist_ok=True)
            df.to_json(preprocessed_path, orient="records", lines=True, force_ascii=False)
            log.info(f"[{bench.bench_name}] Preprocessed data written to {preprocessed_path}")
        except Exception as e:
            log.error(f"[{bench.bench_name}] 预处理失败: {e}")
            log.error(traceback.format_exc())

        storage = FileStorage(
            first_entry_file_name=preprocessed_path,
            cache_path=step_cache_dir,
            file_name_prefix="step",
            cache_type="jsonl",
        )
        
        # 提取关键字段名
        q_key = key_mapping.get("input_question_key")
        ctx_key = key_mapping.get("input_context_key")
        
        # Target keys 处理
        target_key = key_mapping.get("input_target_key")
        targets_key = key_mapping.get("input_targets_key")
        choices_key = key_mapping.get("input_choices_key")
        
        # 强制 choices_key 为 string（如果它是 list）
        if isinstance(choices_key, list):
            # 如果预处理中的合并失败了（比如列不存在），我们只能取第一个作为最后的挣扎，或者直接报错
            # 这里选择保留之前的防御逻辑，但加上警告，表明这是不正常的状态
            log.warning(f"[{bench.bench_name}] input_choices_key is still list {choices_key} after preprocessing. Using first element.")
            choices_key = choices_key[0]

        label_key = key_mapping.get("input_label_key")
        labels_key = key_mapping.get("input_labels_key")
        better_key = key_mapping.get("input_better_key")
        rejected_key = key_mapping.get("input_rejected_key")
        text_key = key_mapping.get("input_text_key")

        # 5. Step 1: Generator
        # 对于不需要生成的任务（如 text_score, choices_a_ll），Generator 可能只是透传或计算
        # BenchAnswerGenerator 内部会根据 eval_type 判断是否需要 generate
        
        # 构造 Prompt Template (简单通用版)
        # 注意：对于 chat 模型，通常建议使用 apply_chat_template，这里简化为 FormatStrPrompt
        # 如果是 base 模型，这个 template 很重要
        prompt_tmpl = FormatStrPrompt(f_str_template="{{question}}\nAnswer:")
        
        generator = BenchAnswerGenerator(
            llm_serving=llm_serving,
            eval_type=bench.bench_dataflow_eval_type,
            prompt_template=prompt_tmpl,
            allow_overwrite=False,
            force_generate=False, # 让算子自己决定
        )

        log.info(f"[{bench.bench_name}] Running Step 1: Generator ({bench.bench_dataflow_eval_type})")
        total_rows = self._count_jsonl_rows(input_dataset_path)
        _emit("generator", generated=0, total=total_rows, percent=0.0)
        step1_output_path = os.path.join(step_cache_dir, "step_step1.jsonl")
        try:
            step1_result: Dict[str, Any] = {"err": None}
            def _run_step1():
                try:
                    generator.run(
                        storage=storage.step(),
                        input_question_key=q_key,
                        input_context_key=ctx_key,
                        input_text_key=text_key,
                        input_choices_key=choices_key,
                        output_key="generated_ans",
                    )
                except Exception as ex:
                    step1_result["err"] = ex
            th = threading.Thread(target=_run_step1, daemon=True)
            th.start()
            last_generated = -1
            while th.is_alive():
                generated = self._count_jsonl_rows(step1_output_path)
                if generated != last_generated:
                    pct = (float(generated) / float(total_rows) * 100.0) if total_rows > 0 else 0.0
                    if pct > 99.0:
                        pct = 99.0
                    _emit("generator", generated=generated, total=total_rows, percent=pct)
                    last_generated = generated
                time.sleep(0.5)
            th.join()
            if step1_result["err"] is not None:
                raise step1_result["err"]
            generated_done = self._count_jsonl_rows(step1_output_path)
            final_pct = 100.0 if total_rows > 0 else 0.0
            _emit("generator", generated=generated_done, total=total_rows, percent=final_pct)
        except Exception as e:
            log.error(f"[{bench.bench_name}] Generator failed: {e}")
            log.error(traceback.format_exc())
            # 强制重置 serving，防止脏状态
            llm_serving = None
            raise e

        # 6. Step 2: Evaluator
        # 先检查 step1 输出中是否包含 target_key 列（参考答案列）
        # 如果数据集没有参考答案（如纯生成任务），跳过评估步骤
        skip_evaluator = False
        required_target_keys = [k for k in [target_key, targets_key, label_key, labels_key, better_key] if k]
        if required_target_keys:
            try:
                step1_df = pd.read_json(step1_output_path, lines=True)
                missing_target_keys = [k for k in required_target_keys if k not in step1_df.columns]
                if missing_target_keys:
                    log.warning(
                        f"[{bench.bench_name}] 数据集缺少参考答案列 {missing_target_keys}，"
                        f"跳过评估步骤（仅保留生成结果）"
                    )
                    skip_evaluator = True
            except Exception as e:
                log.warning(f"[{bench.bench_name}] 无法读取 step1 输出来检查列: {e}")

        if skip_evaluator:
            log.info(f"[{bench.bench_name}] 跳过 Evaluator（无参考答案）")
            _emit("evaluator", generated=total_rows, total=total_rows, percent=100.0)
        else:
            evaluator = UnifiedBenchDatasetEvaluator(
                eval_result_path=eval_result_path,
                llm_serving=llm_serving,
                eval_type=bench.bench_dataflow_eval_type,
                prompt_template=None,
                use_semantic_judge=False,
                metric_type=None,
            )

            log.info(f"[{bench.bench_name}] Running Step 2: Evaluator")
            _emit("evaluator", generated=total_rows, total=total_rows, percent=100.0)

            # 收集所有可能的 input keys
            eval_kwargs = {
                "storage": storage.step(),
                "input_question_key": q_key,
                "input_context_key": ctx_key,
                "input_pred_key": "generated_ans",
                "input_text_key": text_key,
                "input_target_key": target_key,
                "input_targets_key": targets_key,
                "input_choices_key": choices_key,
                "input_label_key": label_key,
                "input_labels_key": labels_key,
                "input_better_key": better_key,
                "input_rejected_key": rejected_key,
            }
            # 过滤 None 和 空字符串
            eval_kwargs = {k: v for k, v in eval_kwargs.items() if v}

            try:
                evaluator.run(**eval_kwargs)
            except Exception as e:
                log.error(f"[{bench.bench_name}] Evaluator failed: {e}")
                log.error(traceback.format_exc())
                llm_serving = None
                raise e

        # 7. 获取结果
        if skip_evaluator:
            # 无参考答案时，step1 输出就是最终结果
            last_step_file = step1_output_path
            stats = {}
            log.info(f"[{bench.bench_name}] 仅生成模式（无参考答案），stats 为空")
        else:
            # step2 产生的文件是包含完整数据的
            files = sorted([f for f in os.listdir(step_cache_dir) if f.endswith(".jsonl") and f.startswith("step_")])
            if not files:
                raise RuntimeError("No step files generated")
            last_step_file = os.path.join(step_cache_dir, files[-1])

            # 读取统计结果
            stats = {}
            if os.path.exists(eval_result_path):
                try:
                    stats_df = pd.read_json(eval_result_path)
                    if not stats_df.empty:
                        stats = stats_df.iloc[0].to_dict()
                except Exception as e:
                    log.error(f"Failed to read stats from {eval_result_path}: {e}")

        return {
            "stats": stats,
            "detail_path": str(Path(last_step_file).absolute()),
            "key_mapping": key_mapping,
        }
