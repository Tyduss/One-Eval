from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from one_eval.logger import get_logger

log = get_logger("ProgressStore")

_LOCK = Lock()
_PROGRESS: Dict[str, Dict[str, Any]] = {}
# Per-thread subscribers; each entry is (event_loop, queue). set_progress will
# fan out via loop.call_soon_threadsafe so worker threads can wake SSE handlers.
_SUBSCRIBERS: Dict[str, List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
_QUEUE_MAXSIZE = 1000


def _parent_thread_id(thread_id: str) -> str:
    """Strip ":model_name" suffix so {thread_id}:{model} updates fan to thread_id subscribers."""
    return thread_id.split(":", 1)[0]


def _safe_put(queue: "asyncio.Queue", payload: Dict[str, Any]) -> None:
    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        log.warning("subscriber queue full, dropping progress payload")


def set_progress(thread_id: str, payload: Dict[str, Any]) -> None:
    """设置进度（支持带模型后缀的 key，如 {thread_id}:{model_name}）。
    会同时广播给该 thread_id 的所有 SSE 订阅者。
    """
    if not thread_id:
        return
    snapshot = dict(payload or {})
    with _LOCK:
        _PROGRESS[thread_id] = snapshot
        subs = list(_SUBSCRIBERS.get(_parent_thread_id(thread_id), []))

    for loop, queue in subs:
        try:
            loop.call_soon_threadsafe(_safe_put, queue, snapshot)
        except RuntimeError:
            # Loop closed; subscriber will be reaped when its handler exits via unsubscribe.
            pass


def get_progress(thread_id: str) -> List[Dict[str, Any]]:
    """获取 thread_id 下的所有进度（包括 {thread_id}:{model_name} 子 key）"""
    if not thread_id:
        return []
    prefix = thread_id + ":"
    with _LOCK:
        results: List[Dict[str, Any]] = []
        val = _PROGRESS.get(thread_id)
        if isinstance(val, dict):
            results.append(dict(val))
        for key, val in _PROGRESS.items():
            if key.startswith(prefix) and isinstance(val, dict):
                results.append(dict(val))
        return results


def clear_progress(thread_id: str) -> None:
    """清除 thread_id 下的所有进度"""
    if not thread_id:
        return
    prefix = thread_id + ":"
    with _LOCK:
        _PROGRESS.pop(thread_id, None)
        keys_to_remove = [k for k in _PROGRESS if k.startswith(prefix)]
        for k in keys_to_remove:
            del _PROGRESS[k]


def subscribe(thread_id: str) -> "asyncio.Queue":
    """SSE handler registers a subscription. Must be called from inside an asyncio
    coroutine so we can capture the running loop."""
    if not thread_id:
        raise ValueError("thread_id required")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    with _LOCK:
        _SUBSCRIBERS.setdefault(thread_id, []).append((loop, queue))
    return queue


def unsubscribe(thread_id: str, queue: "asyncio.Queue") -> None:
    if not thread_id:
        return
    with _LOCK:
        subs = _SUBSCRIBERS.get(thread_id)
        if not subs:
            return
        remaining = [(l, q) for (l, q) in subs if q is not queue]
        if remaining:
            _SUBSCRIBERS[thread_id] = remaining
        else:
            _SUBSCRIBERS.pop(thread_id, None)


def has_subscribers(thread_id: str) -> bool:
    if not thread_id:
        return False
    with _LOCK:
        return bool(_SUBSCRIBERS.get(thread_id))
