from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from one_eval.logger import get_logger

log = get_logger("WorkflowMetaStore")

# Mirror DATA_DIR resolution from one_eval/server/app.py (parents[1] == one_eval,
# parents[1].parent / "server/_data"). We avoid importing from app.py to dodge a
# circular dependency at module-load time.
_THIS_DIR = Path(__file__).resolve().parent  # one_eval/runtime
_SERVER_DATA_DIR = _THIS_DIR.parent / "server" / "_data"
_SERVER_DATA_DIR.mkdir(parents=True, exist_ok=True)

WORKFLOW_META_FILE = _SERVER_DATA_DIR / "workflow_meta.json"

_LOCK = threading.RLock()
_META: Dict[str, Dict[str, Any]] = {}
_DEBOUNCE_TIMER: Optional[threading.Timer] = None
_DEBOUNCE_SECONDS = 1.0
_LOADED = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_workflow_meta() -> None:
    """Restore meta from disk. Call once at process start."""
    global _META, _LOADED
    with _LOCK:
        if _LOADED:
            return
        if WORKFLOW_META_FILE.exists():
            try:
                data = json.loads(WORKFLOW_META_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _META = data
                    # Tag any thread still flagged "running" at process restart as orphaned.
                    for tid, entry in list(_META.items()):
                        if isinstance(entry, dict) and entry.get("status") == "running":
                            entry["status"] = "orphaned"
                            entry["updated_at"] = _now_iso()
                    log.info(f"Loaded workflow_meta from {WORKFLOW_META_FILE} ({len(_META)} threads)")
            except Exception:
                log.error(f"Failed to load {WORKFLOW_META_FILE}", exc_info=True)
                _META = {}
        _LOADED = True


def save_workflow_meta() -> None:
    """Synchronous write to disk. Caller already holds _LOCK or accepts a small race."""
    try:
        with _LOCK:
            snapshot = json.dumps(_META, ensure_ascii=False, indent=2)
        WORKFLOW_META_FILE.write_text(snapshot, encoding="utf-8")
    except Exception:
        log.error(f"Failed to save {WORKFLOW_META_FILE}", exc_info=True)


def _schedule_save() -> None:
    """Debounce: collapse N updates within _DEBOUNCE_SECONDS into a single disk write."""
    global _DEBOUNCE_TIMER
    with _LOCK:
        if _DEBOUNCE_TIMER is not None:
            _DEBOUNCE_TIMER.cancel()
        _DEBOUNCE_TIMER = threading.Timer(_DEBOUNCE_SECONDS, save_workflow_meta)
        _DEBOUNCE_TIMER.daemon = True
        _DEBOUNCE_TIMER.start()


def _ensure_thread(thread_id: str) -> Dict[str, Any]:
    entry = _META.get(thread_id)
    if entry is None:
        entry = {
            "thread_id": thread_id,
            "status": "running",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "cancel_reason": None,
            "benches": {},
        }
        _META[thread_id] = entry
    return entry


def init_thread(thread_id: str) -> None:
    if not thread_id:
        return
    with _LOCK:
        entry = _ensure_thread(thread_id)
        entry["status"] = "running"
        entry["started_at"] = entry.get("started_at") or _now_iso()
        entry["updated_at"] = _now_iso()
    _schedule_save()


def mark_thread(thread_id: str, status: str, error: Optional[str] = None, cancel_reason: Optional[str] = None) -> None:
    if not thread_id:
        return
    with _LOCK:
        entry = _ensure_thread(thread_id)
        entry["status"] = status
        entry["updated_at"] = _now_iso()
        if error is not None:
            entry["error"] = error
        if cancel_reason is not None:
            entry["cancel_reason"] = cancel_reason
    _schedule_save()


def update_model(thread_id: str, bench: str, model: str, payload: Dict[str, Any]) -> None:
    """Merge per-(bench, model) payload. payload may carry status/stage/percent/error/etc."""
    if not thread_id or not bench or not model:
        return
    with _LOCK:
        entry = _ensure_thread(thread_id)
        benches = entry.setdefault("benches", {})
        bench_entry = benches.setdefault(bench, {"models": {}})
        models = bench_entry.setdefault("models", {})
        slot = models.setdefault(model, {})
        # Stamp started_at on first sighting.
        if "started_at" not in slot:
            slot["started_at"] = _now_iso()
        slot.update(payload)
        # If terminal, stamp finished_at.
        if payload.get("status") in ("success", "failed", "cancelled") and "finished_at" not in slot:
            slot["finished_at"] = _now_iso()
        entry["updated_at"] = _now_iso()
    _schedule_save()


def update_workflow_progress(thread_id: str, payload: Dict[str, Any]) -> None:
    """Convenience wrapper: extract bench/model from payload and route to update_model."""
    if not thread_id or not isinstance(payload, dict):
        return
    bench = payload.get("bench_name") or payload.get("bench")
    model = payload.get("model_name") or payload.get("model")
    if not bench or not model:
        return
    slim = {
        k: v
        for k, v in payload.items()
        if k in ("status", "stage", "generated", "total", "percent", "error")
    }
    update_model(thread_id, bench, model, slim)


def get_thread_meta(thread_id: str) -> Optional[Dict[str, Any]]:
    if not thread_id:
        return None
    with _LOCK:
        entry = _META.get(thread_id)
        return json.loads(json.dumps(entry)) if entry is not None else None
