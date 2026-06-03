from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from one_eval.logger import get_logger

log = get_logger("TaskRegistry")


@dataclass
class EvalTaskContext:
    thread_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    model_cancel_events: Dict[str, threading.Event] = field(default_factory=dict)
    cancel_reason: Optional[str] = None
    model_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


_TASK_CTX: Dict[str, EvalTaskContext] = {}
_LOCK = threading.Lock()


def _model_key(bench: str, model: str) -> str:
    return f"{bench}::{model}"


def register_task(thread_id: str) -> EvalTaskContext:
    if not thread_id:
        raise ValueError("thread_id required")
    with _LOCK:
        ctx = _TASK_CTX.get(thread_id)
        if ctx is None:
            ctx = EvalTaskContext(thread_id=thread_id)
            _TASK_CTX[thread_id] = ctx
            log.info(f"Registered task ctx for thread={thread_id}")
        return ctx


def unregister_task(thread_id: str) -> None:
    with _LOCK:
        ctx = _TASK_CTX.pop(thread_id, None)
    if ctx is not None:
        log.info(f"Unregistered task ctx for thread={thread_id}")


def get_ctx(thread_id: str) -> Optional[EvalTaskContext]:
    if not thread_id:
        return None
    with _LOCK:
        return _TASK_CTX.get(thread_id)


def cancel_thread(thread_id: str, reason: str) -> bool:
    """Set global cancel_event for this thread. Returns True if context found."""
    ctx = get_ctx(thread_id)
    if ctx is None:
        return False
    if not ctx.cancel_event.is_set():
        ctx.cancel_reason = reason
        ctx.cancel_event.set()
        log.warning(f"cancel_thread thread={thread_id} reason={reason}")
    return True


def cancel_model(thread_id: str, bench: str, model: str, reason: str) -> bool:
    """Set per-model cancel event so only one (bench, model) is signaled."""
    ctx = get_ctx(thread_id)
    if ctx is None:
        return False
    key = _model_key(bench, model)
    ev = ctx.model_cancel_events.get(key)
    if ev is None:
        ev = threading.Event()
        ctx.model_cancel_events[key] = ev
    if not ev.is_set():
        ev.set()
        log.warning(f"cancel_model thread={thread_id} bench={bench} model={model} reason={reason}")
        # Stash reason on the status dict too so consumers can see it.
        st = ctx.model_status.setdefault(key, {})
        st.setdefault("cancel_reason", reason)
    return True


def is_cancelled(thread_id: str, bench: Optional[str] = None, model: Optional[str] = None) -> bool:
    """True if global cancel_event is set, or if per-model cancel is set for this (bench, model)."""
    ctx = get_ctx(thread_id)
    if ctx is None:
        return False
    if ctx.cancel_event.is_set():
        return True
    if bench and model:
        ev = ctx.model_cancel_events.get(_model_key(bench, model))
        if ev is not None and ev.is_set():
            return True
    return False


def record_model_status(
    thread_id: str,
    bench: str,
    model: str,
    status: str,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    """Mirror per-model status into the registry. Also forwards to workflow_meta_store if available."""
    ctx = get_ctx(thread_id)
    if ctx is not None:
        key = _model_key(bench, model)
        slot = ctx.model_status.setdefault(key, {})
        slot["status"] = status
        if error is not None:
            slot["error"] = error
        for k, v in extra.items():
            slot[k] = v

    # Best-effort mirror into workflow_meta_store (organic dependency; if missing just skip).
    try:
        from one_eval.runtime import workflow_meta_store  # local import to avoid cycle at module load

        payload: Dict[str, Any] = {"status": status}
        if error is not None:
            payload["error"] = error
        payload.update(extra)
        workflow_meta_store.update_model(thread_id, bench, model, payload)
    except Exception:
        pass
