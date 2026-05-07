# one_eval/judges/llm_judge.py
"""核心裁判引擎 — 调用裁判模型对模型输出进行评分。"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from one_eval.core.state import ModelConfig
from one_eval.judges.answer_parser import parse_answer, ParsedAnswer
from one_eval.judges.prompt_builder import build_judge_messages
from one_eval.judges.score_parser import parse_judge_output, ParsedScore
from one_eval.serving.custom_llm_caller import CustomLLMCaller
from one_eval.logger import get_logger

log = get_logger("LLMJudge")


class MockState:
    """Minimal state stub for CustomLLMCaller initialization."""
    def __init__(self, model_name: str):
        self.request = type("R", (), {"model": model_name})()


@dataclass
class JudgeResult:
    """单条样本的裁判评分结果。"""
    sample_index: int
    question: str
    model_name: str
    think: Optional[str] = None
    body: str = ""
    score: Optional[ParsedScore] = None
    judge_time_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_index": self.sample_index,
            "question": self.question,
            "model_name": self.model_name,
        }
        if self.think:
            d["think"] = self.think
        d["body"] = self.body
        if self.score:
            d.update(self.score.to_dict())
        if self.judge_time_ms:
            d["judge_time_ms"] = self.judge_time_ms
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class JudgeTaskConfig:
    """裁判评分任务的配置。"""
    task_id: str
    bench_name: str
    scoring_prompt: str
    judge_model: ModelConfig
    model_names: List[str]             # 待评分的模型列表
    model_data_paths: Dict[str, str]   # {model_name: step1_jsonl_path}
    question_key: str = "question"
    answer_key: str = "generated_ans"
    context_key: Optional[str] = None
    concurrency: int = 5
    output_dir: str = "cache/judge_results"


class LLMJudge:
    """裁判引擎 — 对模型输出进行 LLM-as-Judge 评分。"""

    def __init__(self, config: JudgeTaskConfig):
        self.config = config
        self._caller: Optional[CustomLLMCaller] = None
        self._output_dir = Path(config.output_dir) / config.task_id
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_caller(self) -> CustomLLMCaller:
        """懒初始化 CustomLLMCaller。"""
        if self._caller is None:
            jc = self.config.judge_model
            api_url = jc.api_url or ""
            if api_url and not api_url.endswith("/v1"):
                api_url = api_url.rstrip("/") + "/v1"
            self._caller = CustomLLMCaller(
                state=MockState(jc.model_name_or_path),
                tool_manager=None,
                agent_role="llm_judge",
                model_name=jc.model_name_or_path,
                base_url=api_url,
                api_key=jc.api_key or "",
                api_provider=getattr(jc, "api_provider", "openai_compatible"),
                temperature=0.0,
                request_timeout=getattr(jc, "api_timeout", None),
            )
        return self._caller

    async def judge_single(
        self, question: str, answer: str, model_name: str,
        context: Optional[str] = None,
    ) -> JudgeResult:
        """对单条样本进行裁判评分。"""
        parsed = parse_answer(answer)
        messages = build_judge_messages(
            scoring_prompt=self.config.scoring_prompt,
            question=question,
            body=parsed.body,
            think=parsed.think,
            context=context,
        )
        caller = self._get_caller()
        t0 = time.monotonic()
        try:
            response = await caller.call(messages, bind_post_tools=False)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            score = parse_judge_output(response.content)
            return JudgeResult(
                sample_index=0,  # caller fills this
                question=question,
                model_name=model_name,
                think=parsed.think,
                body=parsed.body,
                score=score,
                judge_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            log.error(f"Judge call failed for model={model_name}: {e}")
            return JudgeResult(
                sample_index=0,
                question=question,
                model_name=model_name,
                think=parsed.think,
                body=parsed.body,
                judge_time_ms=elapsed_ms,
                error=str(e),
            )

    async def judge_model_batch(
        self,
        model_name: str,
        data_path: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[JudgeResult]:
        """对一个模型的所有样本进行批量裁判评分。"""
        if not os.path.exists(data_path):
            log.error(f"Data file not found: {data_path}")
            return []

        df = pd.read_json(data_path, lines=True)
        q_key = self.config.question_key
        a_key = self.config.answer_key
        c_key = self.config.context_key

        if q_key not in df.columns:
            log.error(f"Question key '{q_key}' not in columns: {list(df.columns)}")
            return []
        if a_key not in df.columns:
            log.error(f"Answer key '{a_key}' not in columns: {list(df.columns)}")
            return []

        total = len(df)
        results: List[JudgeResult] = []
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def _judge_one(idx: int, row: pd.Series):
            async with semaphore:
                question = str(row[q_key])
                answer = str(row[a_key])
                context = str(row[c_key]) if c_key and c_key in df.columns else None
                result = await self.judge_single(question, answer, model_name, context)
                result.sample_index = idx
                return result

        # 分批执行以控制并发
        tasks = [_judge_one(i, row) for i, row in df.iterrows()]
        completed = 0

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                log.error(f"Unexpected error in judge_task: {e}")
            completed += 1
            if progress_cb:
                progress_cb({
                    "model_name": model_name,
                    "judged": completed,
                    "total": total,
                    "percent": round(completed / total * 100, 1),
                })

        # 按 sample_index 排序
        results.sort(key=lambda r: r.sample_index)
        return results

    async def judge_all_models(
        self,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, List[JudgeResult]]:
        """对所有待评分模型执行裁判评分。"""
        all_results: Dict[str, List[JudgeResult]] = {}

        for model_name in self.config.model_names:
            data_path = self.config.model_data_paths.get(model_name)
            if not data_path or not os.path.exists(data_path):
                log.warning(f"Skip model {model_name}: data path not found")
                continue

            log.info(f"[{model_name}] Starting judge evaluation...")
            if progress_cb:
                progress_cb({
                    "model_name": model_name,
                    "status": "running",
                    "judged": 0,
                    "total": 0,
                    "percent": 0.0,
                })

            def _model_progress(p: Dict[str, Any]):
                if progress_cb:
                    progress_cb(p)

            results = await self.judge_model_batch(
                model_name, data_path, progress_cb=_model_progress
            )
            all_results[model_name] = results
            log.info(f"[{model_name}] Judge done: {len(results)} samples")

            if progress_cb:
                progress_cb({
                    "model_name": model_name,
                    "status": "done",
                    "judged": len(results),
                    "total": len(results),
                    "percent": 100.0,
                })

        return all_results

    def save_results(
        self, all_results: Dict[str, List[JudgeResult]]
    ) -> Dict[str, str]:
        """保存评分结果到磁盘。返回 {model_name: detail_path}。"""
        saved: Dict[str, str] = {}

        for model_name, results in all_results.items():
            # 明细 JSONL
            detail_path = str(self._output_dir / f"{model_name}_detail.jsonl")
            with open(detail_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            saved[model_name] = detail_path

        # 合并明细（所有模型合并到一个文件，便于对比）
        merged_path = str(self._output_dir / "all_models_detail.jsonl")
        with open(merged_path, "w", encoding="utf-8") as f:
            for model_name, results in all_results.items():
                for r in results:
                    d = r.to_dict()
                    d["model_name"] = model_name
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        saved["_merged"] = merged_path

        return saved
