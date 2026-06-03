from __future__ import annotations

import os
import time
import traceback
import json
import threading
from dataclasses import dataclass, field
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
from one_eval.runtime.task_registry import (
    EvalTaskContext,
    cancel_model,
    is_cancelled,
)

log = get_logger("DataFlowEvalTool")


class AuthError(Exception):
    """API 返回 401/403：API key 或权限有问题。立刻让该 (bench, model) 失败，
    不再走 dataflow 上游的指数退避重试循环。"""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        body_short = (body or "")[:200]
        super().__init__(f"auth_error: HTTP {status_code}: {body_short}")


class CancelledByUserError(Exception):
    """task_ctx 收到 cancel 信号（user stop 或同 (bench, model) 已被标记为失败），
    worker 协作退出。"""


class BatchFatalError(Exception):
    """批次级致命错误。一个 batch 内连续 `max_failures` 次 prompt 失败，
    任何后续 prompt 立即短路返回。watcher 会清理脏 step_step1.jsonl 并向上层 raise。

    注意：dataflow 上游 `_call_generate` / `_run_threadpool` 都有 `except Exception`
    会吞普通异常。BatchFatalError 单纯靠 raise 出不去——必须配合 `_BatchStatus.fatal_flag`
    主动让 watcher 检测。
    """


@dataclass
class _BatchStatus:
    """Per-Serving 批次级失败计数器。

    核心规则：
    - 任意一次 prompt 失败（401/403/timeout/空响应/连接错误等）→ consecutive_failures += 1
    - 任意一次 prompt 成功（拿到非空响应）→ consecutive_failures = 0
    - consecutive_failures >= max_failures → fatal_flag = True，后续 worker 立即短路

    on_update 回调每次 record 后调用，让 watcher 实时把计数推到 SSE。
    """

    max_failures: int = 3
    consecutive_failures: int = 0
    fatal_flag: bool = False
    fatal_reason: Optional[str] = None
    last_error: Optional[str] = None
    successes: int = 0
    failures: int = 0
    # NOTE: 必须 RLock，不能 Lock。record_failure 内部 with lock 后又调 snapshot()，
    # snapshot() 也 with lock — 普通 Lock 会自死锁，worker thread 永远卡住，
    # 表现为 "BatchStatus.record_failure..." 日志一直缺失、SSE 不推送任何 failure 事件。
    lock: threading.RLock = field(default_factory=threading.RLock)
    on_update: Optional[Callable[[Dict[str, Any]], None]] = None

    def reset(self, max_failures: Optional[int] = None) -> None:
        with self.lock:
            if max_failures is not None:
                self.max_failures = max_failures
            self.consecutive_failures = 0
            self.fatal_flag = False
            self.fatal_reason = None
            self.last_error = None
            self.successes = 0
            self.failures = 0

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "max_failures": self.max_failures,
                "consecutive_failures": self.consecutive_failures,
                "fatal_flag": self.fatal_flag,
                "fatal_reason": self.fatal_reason,
                "last_error": self.last_error,
                "successes": self.successes,
                "failures": self.failures,
            }

    def record_failure(self, reason: str, force_fatal: bool = False) -> bool:
        """记录一次失败。返回 True 表示已达到 fatal 阈值（首次触发）。
        """
        triggered = False
        with self.lock:
            self.consecutive_failures += 1
            self.failures += 1
            self.last_error = reason
            if (force_fatal or self.consecutive_failures >= self.max_failures) and not self.fatal_flag:
                self.fatal_flag = True
                self.fatal_reason = (
                    f"Batch aborted after {self.consecutive_failures} consecutive failures. Last error: {reason}"
                )
                triggered = True
            snap = self.snapshot()
        self._emit(snap)
        return triggered

    def record_success(self) -> None:
        with self.lock:
            self.consecutive_failures = 0
            self.successes += 1
            snap = self.snapshot()
        self._emit(snap)

    def _emit(self, snap: Dict[str, Any]) -> None:
        if self.on_update is None:
            return
        try:
            self.on_update(snap)
        except Exception:
            log.exception("batch_status on_update callback failed")


class _APILLMServingWithTimeout(APILLMServing_request):
    """APILLMServing_request 的子类：
    1. 让 timeout 可由实例配置（上游硬编码 60s/1800s）。
    2. 401/403 立即触发 fatal_flag（不再 sleep 重试，浪费用户资源）。
    3. 加批次级失败计数：任意 prompt 失败 +1、成功清零、累计 max_failures 次整批叫停。
    4. success 严格判定：HTTP 200 但响应为空/None 也算失败（防止脏数据混进 step_step1.jsonl）。
    """

    def __init__(
        self,
        *args,
        request_timeout: float = 30.0,
        thread_id: Optional[str] = None,
        bench_name: Optional[str] = None,
        model_label: Optional[str] = None,
        max_failures: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._request_timeout = float(request_timeout) if request_timeout and request_timeout > 0 else 30.0
        # Cancel-routing identity. None-safe: when missing we behave like the old class.
        self._thread_id = thread_id
        self._bench_name = bench_name
        self._model_label = model_label
        # Per-batch failure tracker.
        self._batch_status = _BatchStatus(max_failures=max(1, int(max_failures)))
        # Set by run_eval before each batch.
        self._on_progress: Optional[Callable[[Dict[str, Any]], None]] = None

    def reset_batch(self, max_failures: Optional[int] = None) -> None:
        """Reset per-batch state before starting a fresh bench×model run.

        Called by run_eval even when serving is reused from cache, so consecutive
        benches/threads don't inherit a dirty counter or a sticky fatal_flag.
        """
        self._batch_status.reset(max_failures=max_failures)

    @property
    def batch_status(self) -> _BatchStatus:
        return self._batch_status

    def _signal_auth_failure(self, status_code: int) -> None:
        """Best-effort broadcast a per-model cancel so other in-flight workers stop ASAP."""
        if self._thread_id and self._bench_name and self._model_label:
            try:
                cancel_model(
                    self._thread_id,
                    self._bench_name,
                    self._model_label,
                    reason=f"auth_error:{status_code}",
                )
            except Exception:
                log.debug("cancel_model broadcast failed", exc_info=True)

    def _record_failure_callback(self, reason: str, force_fatal: bool = False) -> None:
        """Notify outer progress callback so SSE gets a 'failure N/3' tick."""
        self._batch_status.record_failure(reason, force_fatal=force_fatal)

    def api_chat(self, system_info: str, messages: str, model: str):
        import requests as _requests
        import json as _json
        import logging as _logging

        # Fast-path: batch already aborted.
        if self._batch_status.fatal_flag:
            raise BatchFatalError(self._batch_status.fatal_reason or "batch aborted")

        try:
            payload = _json.dumps(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_info},
                        {"role": "user", "content": messages},
                    ],
                    "temperature": 0.0,
                }
            )
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            }
            # NOTE: requests `timeout` 可以传 tuple (connect, read)。
            # connect 一般很快（>5s 说明网络出问题），read 取决于模型推理时间。
            # 用 tuple 形式让 connect 超时短一点（5s），便于 stop 时快速感知。
            response = _requests.post(self.api_url, headers=headers, data=payload, timeout=(5.0, self._request_timeout))
            if response.status_code == 200:
                content = self.format_response(response.json())
                if content is None or (isinstance(content, str) and not content.strip()):
                    self._record_failure_callback(f"empty response (HTTP 200)")
                    return None
                self._batch_status.record_success()
                return content
            if response.status_code in (401, 403):
                self._signal_auth_failure(response.status_code)
                self._record_failure_callback(f"HTTP {response.status_code}", force_fatal=True)
                raise AuthError(response.status_code, response.text)
            _logging.error(f"API request failed with status {response.status_code}: {response.text}")
            self._record_failure_callback(f"HTTP {response.status_code}")
            return None
        except AuthError:
            raise
        except Exception as e:
            _logging.error(f"API request error: {e}")
            self._record_failure_callback(f"exception: {type(e).__name__}")
            return None

    def _api_chat_with_id(self, id, payload, model, is_embedding: bool = False, json_schema: dict = None):
        import requests as _requests
        import json as _json
        import logging as _logging

        # Fast-path: batch already aborted, skip work for remaining prompts immediately.
        # This is what stops dataflow from "spinning for 13 minutes" after 3 failures.
        if self._batch_status.fatal_flag:
            return id, None
        # Cross-model cancel propagation: if user clicked Stop on the workflow
        # OR the (bench, model) was specifically cancelled, exit this prompt immediately.
        # The watcher loop in run_eval also checks this, but the worker can spend
        # up to 30s in the HTTP call below before noticing — this guard lets the
        # remaining prompts in the threadpool exit without firing more requests.
        if self._thread_id and self._bench_name and self._model_label:
            try:
                from one_eval.runtime.task_registry import is_cancelled
                if is_cancelled(self._thread_id, self._bench_name, self._model_label):
                    return id, None
            except Exception:
                pass

        try:
            if is_embedding:
                payload = _json.dumps({"model": model, "input": payload})
            elif json_schema is None:
                payload = _json.dumps({"model": model, "messages": payload})
            else:
                payload = _json.dumps(
                    {
                        "model": model,
                        "messages": payload,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "custom_response", "strict": True, "schema": json_schema},
                        },
                    }
                )
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            }
            # NOTE: requests `timeout` 可以传 tuple (connect, read)。
            # connect 一般很快（>5s 说明网络出问题），read 取决于模型推理时间。
            # 用 tuple 形式让 connect 超时短一点（5s），便于 stop 时快速感知。
            response = _requests.post(self.api_url, headers=headers, data=payload, timeout=(5.0, self._request_timeout))
            if response.status_code == 200:
                content = self.format_response(response.json(), is_embedding)
                # Strict success check: empty/None response counts as failure.
                # Embedding returns [] which is valid (don't apply this rule there).
                if not is_embedding and (content is None or (isinstance(content, str) and not content.strip())):
                    self._record_failure_callback(f"empty response for id={id}")
                    return id, None
                self._batch_status.record_success()
                return id, content
            if response.status_code in (401, 403):
                self._signal_auth_failure(response.status_code)
                self._record_failure_callback(f"HTTP {response.status_code}", force_fatal=True)
                raise AuthError(response.status_code, response.text)
            _logging.error(f"API request failed with status {response.status_code}: {response.text}")
            self._record_failure_callback(f"HTTP {response.status_code} (id={id})")
            return id, None
        except AuthError:
            raise
        except Exception as e:
            _logging.error(f"API request error: {e}")
            self._record_failure_callback(f"exception id={id}: {type(e).__name__}")
            return id, None

    def _api_chat_id_retry(self, id, payload, model, is_embedding: bool = False, json_schema: dict = None):
        """Override parent's exponential-backoff retry. We do NOT retry per-prompt:
        each failure counts toward the batch counter. Reaching max_failures aborts
        the whole batch — important so user doesn't wait `max_retries × prompt_count × sleep`
        when the API is unreachable (401 / 5xx / network).
        """
        if self._batch_status.fatal_flag:
            return id, None
        try:
            return self._api_chat_with_id(id, payload, model, is_embedding, json_schema)
        except AuthError:
            # Already counted as failure in _api_chat_with_id. Re-raise so thread worker exits.
            raise

    def _run_threadpool(self, task_args_list: list, desc: str) -> list:
        """Override parent's threadpool to:
        1. Short-circuit cancel pending futures once batch is fatal (don't dispatch
           all N prompts).
        2. Raise BatchFatalError at the end if batch was fatal — though note this
           gets swallowed by dataflow's `_call_generate` `except Exception`, so
           the **canonical** signal is _batch_status.fatal_flag, which the
           run_eval watcher polls.
        3. Also break early on user_stop or sibling-model cancel via task_registry.
        """
        import concurrent.futures as _cf
        from concurrent.futures import ThreadPoolExecutor

        def _cancelled() -> bool:
            if self._thread_id and self._bench_name and self._model_label:
                try:
                    from one_eval.runtime.task_registry import is_cancelled
                    return is_cancelled(self._thread_id, self._bench_name, self._model_label)
                except Exception:
                    pass
            return False

        responses = [None] * len(task_args_list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._api_chat_id_retry, **t) for t in task_args_list]
            try:
                for future in _cf.as_completed(futures):
                    try:
                        rid, response = future.result()
                        if 0 <= rid < len(responses):
                            responses[rid] = response
                    except AuthError:
                        # Already counted. dataflow parent logs this; we just continue.
                        pass
                    except Exception:
                        self.logger.exception("Worker crashed unexpectedly in threadpool")
                    # Early break on fatal OR external cancel (user_stop / sibling-401 broadcast).
                    if self._batch_status.fatal_flag or _cancelled():
                        break
            finally:
                # Cancel pending futures so we don't waste API quota / time.
                for f in futures:
                    if not f.done():
                        f.cancel()
                # ThreadPoolExecutor's context manager will wait for in-flight workers.
        if self._batch_status.fatal_flag:
            raise BatchFatalError(self._batch_status.fatal_reason or "batch aborted")
        return responses


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

    def _get_or_init_llm_serving(
        self,
        config: ModelConfig,
        *,
        thread_id: Optional[str] = None,
        bench_name: Optional[str] = None,
        model_label: Optional[str] = None,
        max_failures: int = 3,
    ) -> LLMServingABC:
        """获取或初始化 LLM Serving（线程安全，返回实例而非写入 self）

        多模型并行评测时，多个线程同时调用 run_eval，
        如果直接写 self.llm_serving 会导致线程间互相覆盖。
        因此改为返回局部 serving 实例，调用方自行持有引用。

        thread_id/bench_name/model_label 用于 _APILLMServingWithTimeout 的
        per-model cancel 路由：401/403 时会调用 cancel_model(...)，让其他 worker
        在 watcher 循环里看到 is_cancelled() 后退出。即使 serving 是缓存复用的，
        这三个字段每次也会被刷新到最新的 task identity。

        max_failures 是批次级致命阈值：连续 max_failures 次 prompt 失败整批叫停。
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
                # Refresh cancel-routing identity AND reset batch counter so the
                # cached serving doesn't inherit a sticky fatal_flag from a prior run.
                if isinstance(cached, _APILLMServingWithTimeout):
                    cached._thread_id = thread_id
                    cached._bench_name = bench_name
                    cached._model_label = model_label
                    cached.reset_batch(max_failures=max_failures)
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
                    thread_id=thread_id,
                    bench_name=bench_name,
                    model_label=model_label,
                    max_failures=getattr(config, "api_max_retries", 3) or 3,
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

    def run_eval(
        self,
        bench: BenchInfo,
        model_config: ModelConfig,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        task_ctx: Optional[EvalTaskContext] = None,
    ) -> Dict[str, Any]:
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

        thread_id = task_ctx.thread_id if task_ctx is not None else None
        model_label = model_config.model_name_or_path

        def _check_cancel(stage_hint: str = ""):
            if task_ctx is None:
                return
            if is_cancelled(task_ctx.thread_id, bench.bench_name, model_label):
                raise CancelledByUserError(
                    f"[{bench.bench_name}/{model_label}] cancelled before {stage_hint}"
                    if stage_hint
                    else f"[{bench.bench_name}/{model_label}] cancelled"
                )

        _check_cancel("init")

        # 1. 准备 Serving（用局部变量持有，避免多线程并发时 self.llm_serving 被覆盖）
        llm_serving = self._get_or_init_llm_serving(
            model_config,
            thread_id=thread_id,
            bench_name=bench.bench_name,
            model_label=model_label,
        )
        # Wire up serving's batch_status → progress_callback so SSE gets
        # real-time `failure_count/max_failures` updates per prompt.
        if isinstance(llm_serving, _APILLMServingWithTimeout) and progress_callback:
            def _on_batch_update(snap: Dict[str, Any]):
                try:
                    progress_callback({
                        "bench_name": bench.bench_name,
                        "model_name": model_label,
                        "stage": "generator",
                        "batch_status": snap,
                        # Also surface counter at top-level so legacy frontends can read it.
                        "consecutive_failures": snap.get("consecutive_failures", 0),
                        "max_failures": snap.get("max_failures", 3),
                        "last_error": snap.get("last_error"),
                    })
                except Exception:
                    log.exception("progress_callback in _on_batch_update failed")
            llm_serving._batch_status.on_update = _on_batch_update

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
        # Helper: move a (likely partial / all-empty) step_step1.jsonl aside so
        # preview/download don't surface garbage to the user.
        def _quarantine_dirty_step1(reason: str):
            if not os.path.exists(step1_output_path):
                return
            try:
                quarantined = step1_output_path.replace(".jsonl", f".failed-{int(time.time())}.jsonl")
                os.rename(step1_output_path, quarantined)
                log.warning(
                    f"[{bench.bench_name}/{model_label}] step_step1.jsonl moved to {quarantined} "
                    f"({reason})"
                )
            except Exception:
                log.debug("quarantine step_step1 failed", exc_info=True)

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
            cancel_observed = False
            fatal_observed = False
            while th.is_alive():
                # User Cancellation (stop button or sibling-model 401 broadcast).
                if task_ctx is not None and is_cancelled(task_ctx.thread_id, bench.bench_name, model_label):
                    log.info(f"[{bench.bench_name}/{model_label}] generator: cancel signaled, breaking watcher")
                    cancel_observed = True
                    break
                # Batch-level abort: consecutive failures reached threshold.
                if isinstance(llm_serving, _APILLMServingWithTimeout) and llm_serving._batch_status.fatal_flag:
                    log.warning(
                        f"[{bench.bench_name}/{model_label}] generator: batch fatal flag detected, breaking watcher. "
                        f"reason={llm_serving._batch_status.fatal_reason}"
                    )
                    fatal_observed = True
                    break
                generated = self._count_jsonl_rows(step1_output_path)
                if generated != last_generated:
                    pct = (float(generated) / float(total_rows) * 100.0) if total_rows > 0 else 0.0
                    if pct > 99.0:
                        pct = 99.0
                    _emit("generator", generated=generated, total=total_rows, percent=pct)
                    last_generated = generated
                time.sleep(0.5)
            # CRITICAL: even after `while th.is_alive()` exits normally (worker
            # finished quickly), the worker may have hit fatal_flag on its last
            # prompt but dataflow's _call_generate swallowed the BatchFatalError.
            # Re-check fatal_flag here so we still surface failure accurately.
            if not fatal_observed and isinstance(llm_serving, _APILLMServingWithTimeout):
                if llm_serving._batch_status.fatal_flag:
                    log.warning(
                        f"[{bench.bench_name}/{model_label}] generator: worker finished but fatal_flag was set "
                        f"(dataflow swallowed BatchFatalError). Marking fatal. reason={llm_serving._batch_status.fatal_reason}"
                    )
                    fatal_observed = True
                elif llm_serving._batch_status.failures > 0:
                    # Soft warning: some prompts failed but didn't hit threshold.
                    # Generated data may contain empty strings.
                    log.warning(
                        f"[{bench.bench_name}/{model_label}] generator: worker finished with "
                        f"{llm_serving._batch_status.failures} non-fatal failures "
                        f"(successes={llm_serving._batch_status.successes})."
                    )
            # Bounded join: give the worker a few seconds to drain, then give up.
            th.join(timeout=3.0)
            if fatal_observed:
                # Step1 may have written some all-empty responses before we noticed — quarantine it
                # so preview doesn't return garbage. Also broadcast per-model cancel so any concurrent
                # evaluator tasks (in step2) abort too.
                _quarantine_dirty_step1("batch fatal")
                if isinstance(llm_serving, _APILLMServingWithTimeout):
                    reason = llm_serving._batch_status.fatal_reason or "batch aborted"
                else:
                    reason = "batch aborted"
                if task_ctx is not None:
                    cancel_model(task_ctx.thread_id, bench.bench_name, model_label, reason=f"batch_fatal:{reason[:80]}")
                raise BatchFatalError(f"[{bench.bench_name}/{model_label}] {reason}")
            if th.is_alive():
                log.warning(
                    f"[{bench.bench_name}/{model_label}] generator worker did not exit in 3s after cancel; abandoning"
                )
                _quarantine_dirty_step1("bounded join timeout")
                raise CancelledByUserError(
                    f"[{bench.bench_name}/{model_label}] generator abandoned after bounded join timeout"
                )
            if cancel_observed:
                _quarantine_dirty_step1("user cancel")
                raise CancelledByUserError(
                    f"[{bench.bench_name}/{model_label}] generator cancelled by user/auth"
                )
            if step1_result["err"] is not None:
                raise step1_result["err"]
            generated_done = self._count_jsonl_rows(step1_output_path)
            final_pct = 100.0 if total_rows > 0 else 0.0
            _emit("generator", generated=generated_done, total=total_rows, percent=final_pct)
        except (CancelledByUserError, BatchFatalError):
            llm_serving = None
            raise
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
            _emit("evaluator", generated=0, total=total_rows, percent=0.0)
            _check_cancel("evaluator")

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
                step2_result: Dict[str, Any] = {"err": None}

                def _run_step2():
                    try:
                        evaluator.run(**eval_kwargs)
                    except Exception as ex:
                        step2_result["err"] = ex

                th2 = threading.Thread(target=_run_step2, daemon=True)
                th2.start()
                cancel_observed2 = False
                fatal_observed2 = False
                while th2.is_alive():
                    if task_ctx is not None and is_cancelled(task_ctx.thread_id, bench.bench_name, model_label):
                        log.info(f"[{bench.bench_name}/{model_label}] evaluator: cancel signaled, breaking watcher")
                        cancel_observed2 = True
                        break
                    if isinstance(llm_serving, _APILLMServingWithTimeout) and llm_serving._batch_status.fatal_flag:
                        log.warning(
                            f"[{bench.bench_name}/{model_label}] evaluator: batch fatal flag detected, breaking watcher"
                        )
                        fatal_observed2 = True
                        break
                    time.sleep(0.5)
                th2.join(timeout=3.0)
                if fatal_observed2:
                    reason = (
                        llm_serving._batch_status.fatal_reason
                        if isinstance(llm_serving, _APILLMServingWithTimeout)
                        else "batch aborted"
                    )
                    raise BatchFatalError(f"[{bench.bench_name}/{model_label}] evaluator {reason or ''}".strip())
                if th2.is_alive():
                    log.warning(
                        f"[{bench.bench_name}/{model_label}] evaluator worker did not exit in 3s after cancel; abandoning"
                    )
                    raise CancelledByUserError(
                        f"[{bench.bench_name}/{model_label}] evaluator abandoned after bounded join timeout"
                    )
                if cancel_observed2:
                    raise CancelledByUserError(
                        f"[{bench.bench_name}/{model_label}] evaluator cancelled by user/auth"
                    )
                if step2_result["err"] is not None:
                    raise step2_result["err"]
                _emit("evaluator", generated=total_rows, total=total_rows, percent=100.0)
            except (CancelledByUserError, BatchFatalError):
                llm_serving = None
                raise
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
