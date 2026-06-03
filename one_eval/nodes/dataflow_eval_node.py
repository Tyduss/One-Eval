from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

from one_eval.core.node import BaseNode
from one_eval.core.state import NodeState, ModelConfig
from one_eval.toolkits.dataflow_eval_tool import (
    AuthError,
    BatchFatalError,
    CancelledByUserError,
    DataFlowEvalTool,
    _APILLMServingWithTimeout,
)
from one_eval.logger import get_logger
from langgraph.types import Command
from one_eval.runtime.progress_store import set_progress, clear_progress
from one_eval.runtime.task_registry import (
    EvalTaskContext,
    register_task,
    unregister_task,
    record_model_status,
)
from one_eval.runtime import workflow_meta_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_batch_counters(tool: DataFlowEvalTool) -> tuple:
    """Best-effort read consecutive_failures / max_failures from the most recent
    _APILLMServingWithTimeout the tool used. Returns (cf, mf) or (None, None) if
    no serving / not an API serving.
    """
    try:
        cache = getattr(tool, "_cached_llm_servings", {}) or {}
        for cached in cache.values():
            if isinstance(cached, _APILLMServingWithTimeout):
                snap = cached._batch_status.snapshot()
                if snap.get("failures", 0) > 0 or snap.get("fatal_flag"):
                    return snap.get("consecutive_failures"), snap.get("max_failures")
        # Fall through: return first serving's snapshot if any.
        for cached in cache.values():
            if isinstance(cached, _APILLMServingWithTimeout):
                snap = cached._batch_status.snapshot()
                return snap.get("consecutive_failures"), snap.get("max_failures")
    except Exception:
        pass
    return None, None


log = get_logger("DataFlowEvalNode")
VALID_EVAL_TYPES = {
    "key1_text_score",
    "key2_qa",
    "key2_q_ma",
    "key3_q_choices_a",
    "key3_q_choices_as",
    "key3_q_a_rejected",
}


class DataFlowEvalNode(BaseNode):
    """
    Step4: DataFlowEvalNode
    - 遍历 benches
    - 检查 eval_status
    - 准备 ModelConfig（支持多模型）
    - 调用 DataFlowEvalTool（并行执行多模型）
    - 更新状态
    """

    def __init__(self):
        self.name = "DataFlowEvalNode"
        self.logger = log
        # 默认输出目录
        self.output_root = os.path.join(os.getcwd(), "cache", "eval_results")
        self._tool = DataFlowEvalTool(output_root=self.output_root)

    def _get_model_configs(self, state: NodeState) -> List[ModelConfig]:
        """获取模型配置列表，优先使用 target_models，否则回退到 target_model"""
        models = getattr(state, "target_models", None) or []
        if models:
            return models

        # 单模型回退
        if state.target_model:
            return [state.target_model]

        # 兜底：从 target_model_name 推断
        if state.target_model_name:
            self.logger.info(f"Using default config for model: {state.target_model_name}")
            return [ModelConfig(
                model_name_or_path=state.target_model_name,
                is_api=False,
                tensor_parallel_size=1,
                max_tokens=2048
            )]

        return []

    async def run(self, state: NodeState, config: Optional[Any] = None) -> NodeState:
        state.current_node = self.name
        thread_id = None
        try:
            if isinstance(config, dict):
                thread_id = ((config.get("configurable") or {}).get("thread_id"))
        except Exception:
            thread_id = None

        ctx: Optional[EvalTaskContext] = None
        if thread_id:
            ctx = register_task(thread_id)
            workflow_meta_store.init_thread(thread_id)

        try:
            return await self._run_inner(state, config, thread_id, ctx)
        finally:
            if thread_id:
                clear_progress(thread_id)
                unregister_task(thread_id)

    async def _run_inner(
        self,
        state: NodeState,
        config: Optional[Any],
        thread_id: Optional[str],
        ctx: Optional[EvalTaskContext],
    ) -> NodeState:

        benches = getattr(state, "benches", None)
        if not benches:
            self.logger.warning("[DataFlowEvalNode] state.benches 为空")
            return state

        # 1. 获取模型配置列表
        model_configs = self._get_model_configs(state)
        if not model_configs:
            self.logger.error("No target_model(s) found!")
            return state

        is_multi_model = len(model_configs) > 1
        if is_multi_model:
            self.logger.info(f"Multi-model evaluation: {[m.model_name_or_path for m in model_configs]}")

        tool = self._tool
        cursor = int(getattr(state, "eval_cursor", 0) or 0)
        if cursor < 0:
            cursor = 0
        if cursor >= len(benches):
            return state

        bench = benches[cursor]

        # 检查 bench 是否已完成所有模型的评测（success / partial 都视为有历史结果）。
        # Prior success slots 在 partial 重跑时必须保留：rerun_model API 把目标 model 改回 pending，
        # 这里只挑 status != success 的 model 来跑，避免覆盖已经成功的代价。
        if bench.eval_status in ("success", "partial") and bench.meta:
            existing_results = bench.meta.get("eval_results", {})
            pending_models = [
                m for m in model_configs
                if existing_results.get(m.model_name_or_path, {}).get("status") != "success"
            ]
            if not pending_models:
                self.logger.info(f"[{bench.bench_name}] 所有模型已成功（partial 复用时也命中），跳过")
                state.eval_cursor = cursor + 1
                return state
            # 部分 model 还需要跑：使用 pending_models 替换本轮的 model_configs。
            # 这样 rerun_model / partial-resume 都能仅重跑失败/取消/未跑的。
            model_configs = pending_models
            is_multi_model = len(model_configs) > 1
            self.logger.info(
                f"[{bench.bench_name}] 已跳过已成功 model，仅重跑: "
                f"{[m.model_name_or_path for m in model_configs]}"
            )
        elif bench.eval_status == "success" and not is_multi_model:
            # 单模型模式：过去直接走 eval_result 兜底
            if bench.meta.get("eval_result"):
                self.logger.info(f"[{bench.bench_name}] 已评测成功，跳过")
                state.eval_cursor = cursor + 1
                return state

        if not bench.dataset_cache:
            self.logger.warning(f"[{bench.bench_name}] 缺少 dataset_cache，跳过")
            bench.eval_status = "failed"
            approved = list(getattr(state, "approved_warning_ids", []) or [])
            confirm_id = "PreEvalReviewNode_confirm"
            approved = [x for x in approved if x != confirm_id]
            return Command(
                goto="PreEvalReviewNode",
                update={
                    "approved_warning_ids": approved,
                    "waiting_for_human": True,
                    "error_flag": True,
                    "error_msg": f"[{bench.bench_name}] 缺少dataset_cache，请检查下载步骤后重试评测。",
                },
            )

        if not bench.bench_dataflow_eval_type:
            self.logger.warning(f"[{bench.bench_name}] 缺少 eval_type，跳过")
            bench.eval_status = "failed"
            approved = list(getattr(state, "approved_warning_ids", []) or [])
            confirm_id = "PreEvalReviewNode_confirm"
            approved = [x for x in approved if x != confirm_id]
            return Command(
                goto="PreEvalReviewNode",
                update={
                    "approved_warning_ids": approved,
                    "waiting_for_human": True,
                    "error_flag": True,
                    "error_msg": f"[{bench.bench_name}] 缺少eval_type，请修正Key Mapping/任务类型后重试评测。",
                },
            )
        if str(bench.bench_dataflow_eval_type).strip() not in VALID_EVAL_TYPES:
            self.logger.warning(f"[{bench.bench_name}] 跳过不支持的 eval_type={bench.bench_dataflow_eval_type}")
            bench.eval_status = "failed"
            state.eval_cursor = cursor + 1
            return state

        # === 执行评测 ===
        if not bench.meta:
            bench.meta = {}

        # 初始化 eval_results 字典（多模型结果存储）
        if "eval_results" not in bench.meta:
            bench.meta["eval_results"] = {}

        bench.eval_status = "running"
        bench.meta["eval_progress"] = {
            "bench_name": bench.bench_name,
            "stage": "queued",
            "generated": 0,
            "total": 0,
            "percent": 0.0,
            "models": [m.model_name_or_path for m in model_configs],
            "completed_models": [],
        }

        # 并行执行多模型评测
        if is_multi_model:
            results = await self._run_multi_model_eval(
                bench, model_configs, tool, thread_id, ctx
            )
        else:
            # 单模型直接执行
            results = await self._run_single_model_eval(
                bench, model_configs[0], tool, thread_id, ctx
            )

        # === 处理结果（区分 success / failed / cancelled，写 per-model schema） ===
        success_models: list = []
        failed_models: list = []
        cancelled_models: list = []
        first_success_key_mapping: Optional[dict] = None

        for model_name, result in results.items():
            slot = bench.meta["eval_results"].setdefault(model_name, {})
            if result.get("success"):
                success_models.append(model_name)
                slot.update({
                    "status": "success",
                    "stats": result.get("stats", {}),
                    "detail_path": result.get("detail_path"),
                    "finished_at": _now_iso(),
                })
                slot.pop("error", None)
                if first_success_key_mapping is None:
                    first_success_key_mapping = result.get("key_mapping", {})
                # Keep legacy single-model fields populated for backward compatibility.
                if not is_multi_model:
                    bench.meta["eval_result"] = result.get("stats", {})
                    bench.meta["eval_detail_path"] = result.get("detail_path")
            elif result.get("cancelled"):
                cancelled_models.append(model_name)
                slot.update({
                    "status": "cancelled",
                    "error": result.get("error", "cancelled"),
                    "finished_at": _now_iso(),
                })
                # Failure detail (counter + reason) so the UI can show "failures=3/3".
                if result.get("consecutive_failures") is not None:
                    slot["consecutive_failures"] = result["consecutive_failures"]
                if result.get("max_failures") is not None:
                    slot["max_failures"] = result["max_failures"]
                self.logger.warning(f"[{bench.bench_name}] 模型 {model_name} 被取消: {result.get('error')}")
            else:
                failed_models.append(model_name)
                slot.update({
                    "status": "failed",
                    "error": result.get("error", "unknown error"),
                    "finished_at": _now_iso(),
                })
                # Persist per-batch failure stats for downstream display / rerun hint.
                if result.get("consecutive_failures") is not None:
                    slot["consecutive_failures"] = result["consecutive_failures"]
                if result.get("max_failures") is not None:
                    slot["max_failures"] = result["max_failures"]
                self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 评测失败: {result.get('error')}")

            record_model_status(
                thread_id,
                bench.bench_name,
                model_name,
                slot["status"],
                error=slot.get("error"),
            ) if thread_id else None

        # Derive bench-level status: all-success → success; any non-success → failed (partial = some success + some failed).
        if success_models and not failed_models and not cancelled_models:
            bench.eval_status = "success"
            if first_success_key_mapping is not None:
                self._set_key_mapping(bench, first_success_key_mapping)
            bench.meta.pop("eval_error", None)
            bench.meta.pop("eval_per_model_errors", None)
            self.logger.info(f"[{bench.bench_name}] 评测完成")
        elif success_models and (failed_models or cancelled_models):
            # Partial success: emit precise per-model errors so the UI / report can show them.
            bench.eval_status = "partial"
            if first_success_key_mapping is not None:
                self._set_key_mapping(bench, first_success_key_mapping)
            bench.meta["eval_per_model_errors"] = {
                m: bench.meta["eval_results"].get(m, {}).get("error", "")
                for m in (failed_models + cancelled_models)
            }
            bench.meta["eval_error"] = "; ".join(
                f"{m}: {(bench.meta['eval_results'].get(m, {}).get('error') or 'unknown')[:120]}"
                for m in (failed_models + cancelled_models)
            )
        else:
            bench.eval_status = "failed"
            bench.meta["eval_per_model_errors"] = {
                m: bench.meta["eval_results"].get(m, {}).get("error", "")
                for m in (failed_models + cancelled_models)
            }
            bench.meta["eval_error"] = "; ".join(
                f"{m}: {(bench.meta['eval_results'].get(m, {}).get('error') or 'unknown')[:120]}"
                for m in (failed_models + cancelled_models)
            )

        state.eval_cursor = cursor + 1
        return state

    async def _run_single_model_eval(
        self,
        bench,
        model_config: ModelConfig,
        tool,
        thread_id: Optional[str],
        ctx: Optional[EvalTaskContext] = None,
    ) -> dict:
        """执行单模型评测"""
        model_name = model_config.model_name_or_path

        # 初始化 per-model 状态条目
        bench.meta.setdefault("eval_results", {})[model_name] = {
            "status": "running",
            "started_at": _now_iso(),
        }
        if thread_id:
            record_model_status(thread_id, bench.bench_name, model_name, "running")

        def _on_progress(p: dict):
            if not bench.meta:
                bench.meta = {}
            bench.meta["eval_progress"] = p
            if thread_id:
                # Strip the heavy batch_status dict before stuffing into set_progress
                # (wealready mirror it into workflow_meta_store below).
                p_for_store = {k: v for k, v in p.items() if k != "batch_status"}
                set_progress(thread_id, p_for_store)
                meta_payload: dict = {
                    "stage": p.get("stage"),
                    "generated": p.get("generated"),
                    "total": p.get("total"),
                    "percent": p.get("percent"),
                }
                # Forward per-prompt failure counter to the workflow_meta model entry.
                bs = p.get("batch_status")
                if isinstance(bs, dict):
                    meta_payload["consecutive_failures"] = bs.get("consecutive_failures", 0)
                    meta_payload["max_failures"] = bs.get("max_failures", 3)
                    meta_payload["failures"] = bs.get("failures", 0)
                    meta_payload["successes"] = bs.get("successes", 0)
                    if bs.get("last_error"):
                        meta_payload["last_error"] = bs["last_error"]
                workflow_meta_store.update_model(
                    thread_id,
                    bench.bench_name,
                    model_name,
                    meta_payload,
                )

        def _emit_final_failure(
            stage: str,
            error: str,
            status: str = "failed",
            cf: Optional[int] = None,
            mf: Optional[int] = None,
        ):
            """Send a final progress event so SSE frontend can move this model
            out of 'running' display."""
            if not thread_id:
                return
            payload: Dict[str, Any] = {
                "bench_name": bench.bench_name,
                "model_name": model_name,
                "stage": stage,
                "status": status,
                "error": error,
                "percent": 100.0,
            }
            if cf is not None:
                payload["consecutive_failures"] = cf
            if mf is not None:
                payload["max_failures"] = mf
            set_progress(thread_id, payload)

        try:
            self.logger.info(f"[{bench.bench_name}] 开始评测模型: {model_name}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool.run_eval(
                    bench, model_config, progress_callback=_on_progress, task_ctx=ctx
                ),
            )

            if thread_id:
                set_progress(thread_id, {
                    "bench_name": bench.bench_name,
                    "model_name": model_name,
                    "stage": "done",
                    "status": "success",
                    "generated": int((bench.meta.get("eval_progress") or {}).get("generated") or 0),
                    "total": int((bench.meta.get("eval_progress") or {}).get("total") or 0),
                    "percent": 100.0,
                })

            return {model_name: {
                "success": True,
                "stats": result["stats"],
                "detail_path": result["detail_path"],
                "key_mapping": result.get("key_mapping", {}),
            }}
        except AuthError as e:
            self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 认证失败: {e}")
            _emit_final_failure("done", str(e), status="failed", cf=1)
            return {model_name: {"success": False, "error": str(e), "consecutive_failures": 1}}
        except BatchFatalError as e:
            self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 批次失败中止: {e}")
            cf, mf = _read_batch_counters(tool)
            _emit_final_failure("done", str(e), status="failed", cf=cf, mf=mf)
            return {model_name: {"success": False, "error": str(e), "consecutive_failures": cf, "max_failures": mf}}
        except CancelledByUserError as e:
            self.logger.warning(f"[{bench.bench_name}] 模型 {model_name} 被取消: {e}")
            cf, mf = _read_batch_counters(tool)
            _emit_final_failure("done", str(e), status="cancelled", cf=cf, mf=mf)
            return {model_name: {"success": False, "cancelled": True, "error": str(e), "consecutive_failures": cf, "max_failures": mf}}
        except Exception as e:
            self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 评测失败: {e}")
            _emit_final_failure("done", str(e), status="failed")
            return {model_name: {"success": False, "error": str(e)}}

    async def _run_multi_model_eval(
        self,
        bench,
        model_configs: List[ModelConfig],
        tool,
        thread_id: Optional[str],
        ctx: Optional[EvalTaskContext] = None,
    ) -> dict:
        """并行执行多模型评测"""
        self.logger.info(f"[{bench.bench_name}] 并行评测 {len(model_configs)} 个模型")

        async def eval_single(model_config: ModelConfig) -> tuple:
            """单个模型评测的异步任务"""
            model_name = model_config.model_name_or_path

            # 初始化 per-model 状态条目（在并发场景里同样必要）
            bench.meta.setdefault("eval_results", {})[model_name] = {
                "status": "running",
                "started_at": _now_iso(),
            }
            if thread_id:
                record_model_status(thread_id, bench.bench_name, model_name, "running")

            def _on_progress(p: dict):
                # 每个模型独立更新进度，用 {thread_id}:{model_name} 做 key
                if thread_id:
                    p_for_store = {k: v for k, v in p.items() if k != "batch_status"}
                    p_with_model = {**p_for_store, "model_name": model_name}
                    set_progress(f"{thread_id}:{model_name}", p_with_model)
                    meta_payload: dict = {
                        "stage": p.get("stage"),
                        "generated": p.get("generated"),
                        "total": p.get("total"),
                        "percent": p.get("percent"),
                    }
                    bs = p.get("batch_status")
                    if isinstance(bs, dict):
                        meta_payload["consecutive_failures"] = bs.get("consecutive_failures", 0)
                        meta_payload["max_failures"] = bs.get("max_failures", 3)
                        meta_payload["failures"] = bs.get("failures", 0)
                        meta_payload["successes"] = bs.get("successes", 0)
                        if bs.get("last_error"):
                            meta_payload["last_error"] = bs["last_error"]
                    workflow_meta_store.update_model(
                        thread_id,
                        bench.bench_name,
                        model_name,
                        meta_payload,
                    )
                # 同时更新 bench.meta 供前端 state 同步
                if not bench.meta:
                    bench.meta = {}
                bench.meta["eval_progress"] = p

            def _emit_final_failure(
                stage: str,
                error: str,
                status: str = "failed",
                cf: Optional[int] = None,
                mf: Optional[int] = None,
            ):
                if not thread_id:
                    return
                payload: Dict[str, Any] = {
                    "bench_name": bench.bench_name,
                    "model_name": model_name,
                    "stage": stage,
                    "status": status,
                    "error": error,
                    "percent": 100.0,
                }
                if cf is not None:
                    payload["consecutive_failures"] = cf
                if mf is not None:
                    payload["max_failures"] = mf
                set_progress(f"{thread_id}:{model_name}", payload)

            try:
                # 在线程池中运行同步的评测方法
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: tool.run_eval(
                        bench, model_config, progress_callback=_on_progress, task_ctx=ctx
                    ),
                )

                # 更新完成的模型列表
                completed = bench.meta.get("eval_progress", {}).get("completed_models", [])
                completed.append(model_name)
                bench.meta["eval_progress"]["completed_models"] = completed

                # 标记该模型完成
                if thread_id:
                    set_progress(f"{thread_id}:{model_name}", {
                        "bench_name": bench.bench_name,
                        "model_name": model_name,
                        "stage": "done",
                        "status": "success",
                        "generated": int((bench.meta.get("eval_progress") or {}).get("generated") or 0),
                        "total": int((bench.meta.get("eval_progress") or {}).get("total") or 0),
                        "percent": 100.0,
                    })

                return model_name, {
                    "success": True,
                    "stats": result["stats"],
                    "detail_path": result["detail_path"],
                    "key_mapping": result.get("key_mapping", {}),
                }
            except AuthError as e:
                self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 认证失败: {e}")
                _emit_final_failure("done", str(e), status="failed", cf=1)
                return model_name, {"success": False, "error": str(e), "consecutive_failures": 1}
            except BatchFatalError as e:
                self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 批次失败中止: {e}")
                cf, mf = _read_batch_counters(tool)
                _emit_final_failure("done", str(e), status="failed", cf=cf, mf=mf)
                return model_name, {"success": False, "error": str(e), "consecutive_failures": cf, "max_failures": mf}
            except CancelledByUserError as e:
                self.logger.warning(f"[{bench.bench_name}] 模型 {model_name} 被取消: {e}")
                cf, mf = _read_batch_counters(tool)
                _emit_final_failure("done", str(e), status="cancelled", cf=cf, mf=mf)
                return model_name, {"success": False, "cancelled": True, "error": str(e), "consecutive_failures": cf, "max_failures": mf}
            except Exception as e:
                self.logger.error(f"[{bench.bench_name}] 模型 {model_name} 评测失败: {e}")
                _emit_final_failure("done", str(e), status="failed")
                return model_name, {"success": False, "error": str(e)}

        # 并行执行所有模型评测
        tasks = [eval_single(m) for m in model_configs]
        results_list = await asyncio.gather(*tasks)

        # 转换为字典
        results = dict(results_list)

        if thread_id:
            set_progress(thread_id, {
                "bench_name": bench.bench_name,
                "stage": "done",
                "generated": 0,
                "total": 0,
                "percent": 100.0,
                "models": [m.model_name_or_path for m in model_configs],
                "completed_models": list(results.keys()),
            })

        return results

    def _set_key_mapping(self, bench, final_key_mapping: dict):
        """设置 key mapping 信息"""
        eval_type = bench.bench_dataflow_eval_type

        # 确定 Pred Key
        default_pred_key = "generated_ans"
        mapped_pred_key = final_key_mapping.get("input_pred_key")
        pred_key = mapped_pred_key if mapped_pred_key else default_pred_key

        # 确定 Ref Key
        ref_key = None
        if eval_type == "key2_qa":
            ref_key = final_key_mapping.get("input_target_key")
        elif eval_type == "key2_q_ma":
            ref_key = final_key_mapping.get("input_targets_key")
        elif eval_type == "key3_q_choices_a":
            ref_key = final_key_mapping.get("input_label_key")
            pred_key = "eval_pred"
        elif eval_type == "key3_q_choices_as":
            ref_key = final_key_mapping.get("input_labels_key")
            pred_key = "eval_pred"
        elif eval_type == "key3_q_a_rejected":
            ref_key = final_key_mapping.get("input_better_key")
        elif eval_type == "key1_text_score":
            ref_key = None
            if final_key_mapping.get("input_text_key"):
                pred_key = final_key_mapping.get("input_text_key")

        if ref_key:
            bench.meta["ref_key"] = ref_key
            self.logger.info(f"[{bench.bench_name}] Set ref_key='{ref_key}'")

        bench.meta["pred_key"] = pred_key
        self.logger.info(f"[{bench.bench_name}] Set pred_key='{pred_key}'")
