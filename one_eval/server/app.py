
import asyncio
import uuid
import json
import os
import re
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from one_eval.logger import get_logger
from one_eval.toolkits.hf_download_tool import HFDownloadTool
from one_eval.runtime.progress_store import get_progress, clear_progress

log = get_logger("OneEval-Server")

# === Early Environment Setup ===
# Must be done before importing langgraph/transformers/etc. to ensure env vars take effect
SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"
MODELS_FILE = DATA_DIR / "models.json"
THREAD_META_FILE = DATA_DIR / "thread_meta.json"
# SERVER_DIR is .../one_eval/server
# parents[0]=one_eval, parents[1]=One-Eval (Repo Root)
REPO_ROOT = SERVER_DIR.parents[1]
ENV_FILE = REPO_ROOT / "env.sh"

# Original DB location was parents[2] (scy/checkpoints)
# We keep it there or move it? 
# If previous code used parents[2], we should respect it to find existing DB.
DB_PATH = (SERVER_DIR.parents[2] / "checkpoints" / "eval.db").resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 内存缓存：记录每个 thread 是否处于 interrupted 状态
# ainvoke 返回时立即写入，避免 get_status 依赖 checkpoint 竞态
_thread_interrupt_cache: Dict[str, bool] = {}

def _load_env_file():
    """Parse env.sh and set os.environ if not already set."""
    if not ENV_FILE.exists():
        return
    
    log.info(f"Loading env from {ENV_FILE}")
    content = ENV_FILE.read_text()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support 'export KEY=VALUE' or 'KEY=VALUE'
        if line.startswith("export "):
            line = line[7:].strip()
        
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Only set if not already set (allow shell override) or force?
            # User wants to avoid export, so we should set it if missing.
            # But if config.json exists, it might override later.
            if key not in os.environ and val:
                os.environ[key] = val
                log.info(f"Set {key} from env.sh")

_load_env_file()

def _load_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text())
    except Exception:
        log.error(f"Error loading {path}: ", exc_info=True)
        return default

def _write_json_file(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        log.error(f"Error writing {path}: ", exc_info=True)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_thread_meta() -> Dict[str, Any]:
    data = _load_json_file(THREAD_META_FILE, default={})
    return data if isinstance(data, dict) else {}

def _set_thread_created_at(thread_id: str, created_at: Optional[str] = None) -> str:
    ts = created_at or _now_iso()
    meta = _load_thread_meta()
    item = meta.get(thread_id)
    if not isinstance(item, dict):
        item = {}
    item["created_at"] = ts
    item["updated_at"] = ts
    meta[thread_id] = item
    _write_json_file(THREAD_META_FILE, meta)
    return ts

def _touch_thread_updated_at(thread_id: str, updated_at: Optional[str] = None) -> None:
    meta = _load_thread_meta()
    item = meta.get(thread_id)
    if not isinstance(item, dict):
        item = {}
    if "created_at" not in item or not item.get("created_at"):
        item["created_at"] = updated_at or _now_iso()
    item["updated_at"] = updated_at or _now_iso()
    meta[thread_id] = item
    _write_json_file(THREAD_META_FILE, meta)

def _normalize_model_path_for_host(raw: str) -> str:
    p = (raw or "").strip()
    if not p:
        return p
    if os.name == "nt":
        m = re.match(r"^/mnt/([a-zA-Z])/(.+)$", p)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}"
        return p
    m = re.match(r"^([a-zA-Z]):\\(.+)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p

def load_server_config() -> Dict[str, Any]:
    cfg = _load_json_file(CONFIG_FILE, default={})
    if not isinstance(cfg, dict):
        cfg = {}
        
    # Merge env.sh defaults if config is empty
    # (Optional, but good for first run)
    
    hf = cfg.get("hf")
    if not isinstance(hf, dict):
        hf = {}
    endpoint = hf.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        # Fallback to env or default
        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    token = hf.get("token")
    if token is not None and (not isinstance(token, str) or not token.strip()):
        token = None
    # If token missing in config, maybe check env?
    if not token and os.environ.get("HF_TOKEN"):
         token = os.environ.get("HF_TOKEN")
         
    cfg["hf"] = {"endpoint": endpoint, "token": token}

    agent = cfg.get("agent")
    if not isinstance(agent, dict):
        agent = {}
    provider = agent.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        provider = "openai_compatible"
    base_url = agent.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = os.environ.get("DF_API_BASE_URL", "http://123.129.219.111:3000/v1")
    model = agent.get("model")
    if not isinstance(model, str) or not model.strip():
        model = os.environ.get("DF_MODEL_NAME", "gpt-4o")
    api_key = agent.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        api_key = None
    if not api_key and os.environ.get("OE_API_KEY"):
        api_key = os.environ.get("OE_API_KEY")

    agent_timeout_s = agent.get("timeout_s")
    if not isinstance(agent_timeout_s, int) or agent_timeout_s <= 0:
        agent_timeout_s = 15
    cfg["agent"] = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout_s": agent_timeout_s,
    }
    return cfg

def save_server_config(cfg: Dict[str, Any]) -> None:
    _write_json_file(CONFIG_FILE, cfg)

def apply_hf_env_from_config(cfg: Dict[str, Any]) -> None:
    hf = cfg.get("hf") or {}
    endpoint = hf.get("endpoint")
    token = hf.get("token")
    if isinstance(endpoint, str) and endpoint.strip():
        os.environ["HF_ENDPOINT"] = endpoint.strip()
    if isinstance(token, str) and token.strip():
        os.environ["HF_TOKEN"] = token.strip()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token.strip()

def _normalize_openai_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if u.endswith("/v1/chat/completions"):
        u = u[: -len("/v1/chat/completions")] + "/v1"
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    if u.endswith("/v1/"):
        u = u[:-1]
    return u


def _normalize_anthropic_base_url(url: str) -> str:
    """确保 Anthropic URL 以 /v1 结尾（不含尾斜杠）。"""
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    # 去掉已有的路径后缀
    for suffix in ("/v1/messages", "/messages"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    # 确保 /v1 存在
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u

def apply_agent_env_from_config(cfg: Dict[str, Any]) -> None:
    agent = cfg.get("agent") or {}
    base_url = agent.get("base_url")
    api_key = agent.get("api_key")
    model = agent.get("model")
    timeout_s = agent.get("timeout_s")
    provider = agent.get("provider")
    if isinstance(base_url, str) and base_url.strip():
        os.environ["OE_API_BASE"] = _normalize_openai_base_url(base_url.strip())
        os.environ["DF_API_BASE_URL"] = _normalize_openai_base_url(base_url.strip())
    if isinstance(api_key, str) and api_key.strip():
        os.environ["OE_API_KEY"] = api_key.strip()
        os.environ["DF_API_KEY"] = api_key.strip()
    if isinstance(model, str) and model.strip():
        os.environ["DF_MODEL_NAME"] = model.strip()
        os.environ["OE_MODEL_NAME"] = model.strip()
    if isinstance(timeout_s, int) and timeout_s > 0:
        os.environ["OE_TIMEOUT_S"] = str(timeout_s)
        os.environ["DF_TIMEOUT_S"] = str(timeout_s)
    if isinstance(provider, str) and provider.strip():
        os.environ["OE_API_PROVIDER"] = provider.strip()

# Initialize Env ASAP
_cfg0 = load_server_config()
log.info(f"Loaded server config: {_cfg0}")
if not CONFIG_FILE.exists():
    save_server_config(_cfg0)
apply_hf_env_from_config(_cfg0)
apply_agent_env_from_config(_cfg0)

from one_eval.graph.workflow_all import build_complete_workflow
from one_eval.utils.checkpoint import get_checkpointer
from one_eval.core.state import NodeState, ModelConfig, BenchInfo, MainRequest
from one_eval.utils.deal_json import _save_state_json
from langgraph.types import Command
from one_eval.utils.bench_registry import BenchRegistry
from one_eval.core.metric_registry import get_registered_metrics_meta, MetricMeta

# Bench Registry - 使用双文件模式（公共 + 本地映射）
BENCH_GALLERY_PATH = REPO_ROOT / "one_eval" / "utils" / "bench_table" / "bench_gallery.json"
BENCH_GALLERY_PUBLIC_PATH = REPO_ROOT / "one_eval" / "utils" / "bench_table" / "bench_gallery_public.json"
BENCH_GALLERY_LOCAL_PATH = REPO_ROOT / "one_eval" / "utils" / "bench_table" / "bench_gallery_local.json"

# 优先使用双文件模式，如果公共文件不存在则回退到单一文件
if BENCH_GALLERY_PUBLIC_PATH.exists():
    bench_registry = BenchRegistry(
        str(BENCH_GALLERY_PUBLIC_PATH),
        local_mapping_path=str(BENCH_GALLERY_LOCAL_PATH) if BENCH_GALLERY_LOCAL_PATH.exists() else None
    )
else:
    # 回退到旧的单一文件模式（向后兼容）
    bench_registry = BenchRegistry(str(BENCH_GALLERY_PATH))

# Models
class HFConfigResponse(BaseModel):
    endpoint: str
    token_set: bool

class HFConfigUpdateRequest(BaseModel):
    endpoint: Optional[str] = None
    token: Optional[str] = None
    clear_token: bool = False

class HFTestRequest(BaseModel):
    endpoint: Optional[str] = None
    token: Optional[str] = None

class HFTestResponse(BaseModel):
    ok: bool
    status_code: Optional[int] = None
    detail: str
    endpoint: str

class AgentConfigResponse(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_set: bool
    timeout_s: int

class AgentConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False
    timeout_s: Optional[int] = None

class AgentTestResponse(BaseModel):
    ok: bool
    status_code: Optional[int] = None
    detail: str
    mode: str

app = FastAPI(title="One Eval API")
RUNNING_WORKFLOW_TASKS: Dict[str, asyncio.Task] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/config/hf", response_model=HFConfigResponse)
def get_hf_config():
    cfg = load_server_config()
    hf = cfg.get("hf") or {}
    endpoint = hf.get("endpoint") or "https://hf-mirror.com"
    token = hf.get("token")
    return {"endpoint": endpoint, "token_set": isinstance(token, str) and bool(token.strip())}

@app.post("/api/config/hf", response_model=HFConfigResponse)
def update_hf_config(req: HFConfigUpdateRequest):
    cfg = load_server_config()
    hf = cfg.get("hf") or {}
    endpoint = hf.get("endpoint") or "https://hf-mirror.com"
    token = hf.get("token")

    if req.endpoint is not None:
        ep = req.endpoint.strip()
        endpoint = ep if ep else "https://hf-mirror.com"

    if req.clear_token:
        token = None
    elif req.token is not None:
        tk = req.token.strip()
        if tk:
            token = tk

    cfg["hf"] = {"endpoint": endpoint, "token": token}
    save_server_config(cfg)
    apply_hf_env_from_config(cfg)
    return {"endpoint": endpoint, "token_set": isinstance(token, str) and bool(token.strip())}

@app.post("/api/config/hf/test", response_model=HFTestResponse)
async def test_hf_config(req: Optional[HFTestRequest] = None):
    """Test HuggingFace endpoint connectivity"""
    cfg = load_server_config()
    hf = cfg.get("hf") or {}

    # Get endpoint to test
    endpoint = hf.get("endpoint") or "https://hf-mirror.com"
    if req and req.endpoint and req.endpoint.strip():
        endpoint = req.endpoint.strip()

    # Get token (optional)
    token = hf.get("token")
    if req and req.token is not None:
        token = req.token.strip() if req.token.strip() else None

    # Normalize endpoint URL
    endpoint = endpoint.rstrip("/")
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test by fetching the /models endpoint (public API)
            r = await client.get(f"{endpoint}/models", headers=headers)
            if r.status_code == 200:
                return {
                    "ok": True,
                    "status_code": r.status_code,
                    "detail": f"连接成功: {endpoint}",
                    "endpoint": endpoint
                }
            else:
                return {
                    "ok": False,
                    "status_code": r.status_code,
                    "detail": f"HTTP {r.status_code}: {r.text[:200]}",
                    "endpoint": endpoint
                }
        except httpx.TimeoutStatus:
            return {
                "ok": False,
                "status_code": None,
                "detail": f"连接超时: {endpoint}",
                "endpoint": endpoint
            }
        except httpx.ConnectError as e:
            return {
                "ok": False,
                "status_code": None,
                "detail": f"无法连接: {str(e)}",
                "endpoint": endpoint
            }
        except Exception as e:
            return {
                "ok": False,
                "status_code": None,
                "detail": f"测试失败: {str(e)}",
                "endpoint": endpoint
            }

@app.get("/api/config/agent", response_model=AgentConfigResponse)
def get_agent_config():
    cfg = load_server_config()
    agent = cfg.get("agent") or {}
    base_url = _normalize_openai_base_url(agent.get("base_url") or "")
    model = agent.get("model") or "gpt-4o"
    provider = agent.get("provider") or "openai_compatible"
    timeout_s = agent.get("timeout_s") or 15
    api_key = agent.get("api_key")
    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key_set": isinstance(api_key, str) and bool(api_key.strip()),
        "timeout_s": int(timeout_s),
    }

@app.post("/api/config/agent", response_model=AgentConfigResponse)
def update_agent_config(req: AgentConfigUpdateRequest):
    cfg = load_server_config()
    agent = cfg.get("agent") or {}

    provider = agent.get("provider") or "openai_compatible"
    base_url = agent.get("base_url") or "http://123.129.219.111:3000/v1"
    model = agent.get("model") or "gpt-4o"
    api_key = agent.get("api_key")
    timeout_s = agent.get("timeout_s") or 15

    if req.provider is not None and req.provider.strip():
        provider = req.provider.strip()
    if req.base_url is not None and req.base_url.strip():
        base_url = _normalize_openai_base_url(req.base_url.strip())
    if req.model is not None and req.model.strip():
        model = req.model.strip()

    if req.clear_api_key:
        api_key = None
    elif req.api_key is not None:
        k = req.api_key.strip()
        if k:
            api_key = k

    if req.timeout_s is not None:
        if req.timeout_s > 0:
            timeout_s = int(req.timeout_s)

    cfg["agent"] = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "timeout_s": timeout_s,
    }
    save_server_config(cfg)
    apply_agent_env_from_config(cfg)
    return {
        "provider": provider,
        "base_url": _normalize_openai_base_url(base_url),
        "model": model,
        "api_key_set": isinstance(api_key, str) and bool(api_key.strip()),
        "timeout_s": timeout_s,
    }

class AgentTestRequest(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: Optional[int] = None

import httpx

@app.post("/api/config/agent/test", response_model=AgentTestResponse)
async def test_agent_config(req: Optional[AgentTestRequest] = None):
    cfg = load_server_config()
    agent = cfg.get("agent") or {}
    
    base_url = agent.get("base_url") or ""
    if req and req.base_url and req.base_url.strip():
        base_url = req.base_url.strip()

    # 先确定协议类型（从请求或已保存的配置中）
    provider = agent.get("provider") or "openai_compatible"
    if req and req.provider and req.provider.strip():
        provider = req.provider.strip()

    # 根据协议类型使用对应的 URL 规范化
    if provider == "anthropic":
        base_url = _normalize_anthropic_base_url(base_url)
    else:
        base_url = _normalize_openai_base_url(base_url)

    api_key = agent.get("api_key")
    if req and req.api_key is not None:
        api_key = req.api_key.strip()

    model = agent.get("model") or "gpt-4o"
    if req and req.model and req.model.strip():
        model = req.model.strip()

    timeout_s = int(agent.get("timeout_s") or 15)
    if req and req.timeout_s and req.timeout_s > 0:
        timeout_s = req.timeout_s

    if provider == "anthropic":
        # Anthropic Messages API
        headers: Dict[str, str] = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if isinstance(api_key, str) and api_key.strip():
            headers["x-api-key"] = api_key.strip()
        chat_url = f"{base_url}/messages"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }
    else:
        # OpenAI-compatible
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if isinstance(api_key, str) and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        chat_url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
        }

    models_ok = False
    models_status: Optional[int] = None
    models_detail = ""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        # /models 仅对 OpenAI 兼容协议有意义
        if provider != "anthropic":
            try:
                r = await client.get(f"{base_url}/models", headers=headers)
                if r.status_code == 200:
                    models_ok = True
                    models_status = r.status_code
                    models_detail = "GET /models ok"
            except Exception:
                pass

        try:
            r = await client.post(chat_url, headers=headers, json=payload)
            if 200 <= r.status_code < 300:
                endpoint_name = "/messages" if provider == "anthropic" else "/chat/completions"
                detail = f"POST {endpoint_name} ok"
                if models_ok:
                    detail = f"{models_detail}; {detail}"
                return {"ok": True, "status_code": r.status_code, "detail": detail, "mode": "chat"}

            try:
                err_detail = r.json()
            except:
                err_detail = r.text[:200]

            if r.status_code in (401, 403):
                return {"ok": False, "status_code": r.status_code, "detail": f"Unauthorized: {err_detail}", "mode": "chat"}

            return {"ok": False, "status_code": r.status_code, "detail": f"Request failed: {err_detail}", "mode": "chat"}
        except Exception as e:
            if models_ok:
                return {"ok": False, "status_code": models_status, "detail": f"{models_detail}; chat failed: {e}", "mode": "chat"}
            return {"ok": False, "status_code": None, "detail": f"Connection error: {e}", "mode": "chat"}

class StartWorkflowRequest(BaseModel):
    user_query: str
    target_model_name: str
    target_model_path: str
    language: str = "zh"
    tensor_parallel_size: int = 1
    max_tokens: int = 2048
    use_rag: bool = True
    local_count: int = 3
    hf_count: int = 2
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    # API model support
    is_api: bool = False
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    api_provider: str = "openai_compatible"
    # Multi-model support
    target_models: Optional[List[Dict[str, Any]]] = None

class ResumeWorkflowRequest(BaseModel):
    thread_id: str
    action: str = "approved"  # or "rejected", etc.
    feedback: Optional[str] = None
    selected_benches: Optional[List[str]] = None
    state_updates: Optional[Dict[str, Any]] = None # For manual config modifications

class RedownloadBenchRequest(BaseModel):
    bench_name: str
    repo_id: Optional[str] = None
    config: Optional[str] = None
    split: Optional[str] = None
    force: bool = False

class RerunExecutionRequest(BaseModel):
    bench_name: Optional[str] = None
    state_updates: Optional[Dict[str, Any]] = None
    goto_confirm: bool = True

class ManualBenchRequest(BaseModel):
    bench_name: str
    dataset_cache: str
    bench_dataflow_eval_type: str
    meta: Optional[Dict[str, Any]] = None

class ManualStartRequest(BaseModel):
    user_query: str = "manual eval"
    target_model_name: Optional[str] = None
    language: str = "zh"
    target_model: Optional[Dict[str, Any]] = None  # Single model (backward compatible)
    target_models: Optional[List[Dict[str, Any]]] = None  # Multi-model support
    benches: List[ManualBenchRequest]

class WorkflowStatusResponse(BaseModel):
    thread_id: str
    status: str # "running", "interrupted", "completed", "failed", "idle"
    current_node: Optional[str] = None
    state_values: Optional[Dict[str, Any]] = None
    next_node: Optional[str] = None

class HistoryItem(BaseModel):
    thread_id: str
    updated_at: str
    user_query: Optional[str] = None
    status: str

# ... (Previous imports)

@app.post("/api/workflow/start")
async def start_workflow(req: StartWorkflowRequest):
    thread_id = str(uuid.uuid4())
    _set_thread_created_at(thread_id)
    log.info(f"Starting workflow for thread_id={thread_id}")

    # Build ModelConfig list
    target_models_list: List[ModelConfig] = []

    if req.target_models:
        # Multi-model mode
        for m in req.target_models:
            cfg = ModelConfig(
                model_name_or_path=m.get("path", m.get("model_name_or_path", "")),
                is_api=m.get("is_api", False),
                api_url=m.get("api_url"),
                api_key=m.get("api_key"),
                api_provider=m.get("api_provider", "openai_compatible"),
                tensor_parallel_size=m.get("tensor_parallel_size", req.tensor_parallel_size),
                max_tokens=m.get("max_tokens", req.max_tokens),
                temperature=m.get("temperature", req.temperature),
                top_p=m.get("top_p", req.top_p),
                top_k=m.get("top_k", req.top_k),
                repetition_penalty=m.get("repetition_penalty", req.repetition_penalty),
                max_model_len=m.get("max_model_len", req.max_model_len),
                gpu_memory_utilization=m.get("gpu_memory_utilization", req.gpu_memory_utilization),
            )
            target_models_list.append(cfg)
        log.info(f"Multi-model mode: {len(target_models_list)} models")

    # Fallback to single model (backward compatible)
    if not target_models_list and req.target_model_path:
        single_model = ModelConfig(
            model_name_or_path=req.target_model_path,
            is_api=req.is_api,
            api_url=req.api_url,
            api_key=req.api_key,
            api_provider=req.api_provider,
            tensor_parallel_size=req.tensor_parallel_size,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_model_len=req.max_model_len,
            gpu_memory_utilization=req.gpu_memory_utilization,
        )
        target_models_list.append(single_model)
        log.info(f"Single-model mode (fallback)")

    # Initialize State
    initial_state = NodeState(
        user_query=req.user_query,
        target_model_name=req.target_model_name,
        request=MainRequest(language=req.language),
        use_rag=req.use_rag,
        local_count=req.local_count,
        hf_count=req.hf_count,
        target_model=target_models_list[0] if target_models_list else None,
        target_models=target_models_list,
    )

    _launch_graph_task(thread_id, initial_state)

    return {"thread_id": thread_id, "status": "started", "model_count": len(target_models_list)}

async def run_graph_background(thread_id: str, input_state: Any, resume_command: Optional[Command] = None):
    # Ensure env is fresh (though we set it at top level, dynamic updates might need this)
    apply_hf_env_from_config(load_server_config())
    apply_agent_env_from_config(load_server_config())
    
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            _touch_thread_updated_at(thread_id)
            log.info(f"Invoking graph for {thread_id}")
            if resume_command:
                # If resume_command is passed, we assume state updates were handled before calling this if needed
                result = await graph.ainvoke(resume_command, config=config)
            else:
                result = await graph.ainvoke(input_state, config=config)

            # Check if workflow was interrupted
            if result and "__interrupt__" in result:
                log.info(f"Graph interrupted for {thread_id}, interrupts: {result['__interrupt__']}")
                _thread_interrupt_cache[thread_id] = True
                _touch_thread_updated_at(thread_id)
            else:
                log.info(f"Graph execution finished for {thread_id}")
                _thread_interrupt_cache.pop(thread_id, None)
                _touch_thread_updated_at(thread_id)
        except asyncio.CancelledError:
            log.warning(f"Graph execution cancelled by user for {thread_id}")
            _touch_thread_updated_at(thread_id)
            raise
        except Exception as e:
            log.error(f"Error executing graph for {thread_id}: {e}")
            _touch_thread_updated_at(thread_id)
        finally:
            clear_progress(thread_id)
            task = RUNNING_WORKFLOW_TASKS.get(thread_id)
            if task is asyncio.current_task():
                RUNNING_WORKFLOW_TASKS.pop(thread_id, None)

def _launch_graph_task(thread_id: str, input_state: Any = None, resume_command: Optional[Command] = None):
    old = RUNNING_WORKFLOW_TASKS.get(thread_id)
    if old and not old.done():
        log.warning(f"Cancelling existing task for {thread_id} because a new task is being launched. New State: {bool(input_state)}, Resume: {resume_command}")
        old.cancel()
    task = asyncio.create_task(run_graph_background(thread_id, input_state, resume_command=resume_command))
    RUNNING_WORKFLOW_TASKS[thread_id] = task
    return task

@app.post("/api/workflow/stop/{thread_id}")
async def stop_workflow(thread_id: str):
    task = RUNNING_WORKFLOW_TASKS.get(thread_id)
    if not task:
        log.info(f"Stop request for {thread_id}, but no running task found.")
        return {"thread_id": thread_id, "status": "idle", "detail": "no running workflow"}
    if task.done():
        RUNNING_WORKFLOW_TASKS.pop(thread_id, None)
        log.info(f"Stop request for {thread_id}, task already finished.")
        return {"thread_id": thread_id, "status": "idle", "detail": "workflow already finished"}
    
    log.warning(f"Stop request received for {thread_id}. Cancelling task...")
    task.cancel()
    return {"thread_id": thread_id, "status": "stopping"}

@app.get("/api/workflow/status/{thread_id}")
async def get_status(thread_id: str):
    """
    获取工作流状态。

    解决 interrupt() 执行期间的竞态条件：
    当 next=() 且 interrupts=[] 但有 benches 数据时，
    可能是 interrupt() 正在执行中，需要短暂等待并重试。
    """
    import asyncio

    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 竞态重试：interrupt() 写入 checkpoint 需要一点时间
        # 最多重试 8 次，间隔从 100ms 线性增加到 300ms
        max_retries = 8
        retry_delays = [0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3]

        for attempt in range(max_retries):
            try:
                snap = await graph.aget_state(config)
            except Exception as e:
                log.error(f"Failed to get state for {thread_id}: {e}")
                return {"thread_id": thread_id, "status": "not_found"}

            if not snap or (not snap.values and not snap.next):
                return {"thread_id": thread_id, "status": "idle"}

            next_nodes = snap.next
            current_values = snap.values or {}
            interrupts = snap.interrupts

            INTERRUPT_NODES = ("HumanReviewNode", "PreEvalReviewNode", "MetricReviewNode")

            # 检测中断
            is_interrupted = bool(interrupts and len(interrupts) > 0)
            if not is_interrupted and next_nodes:
                is_interrupted = any(node in next_nodes for node in INTERRUPT_NODES)

            # 判断状态
            if not next_nodes and not is_interrupted:
                # 优先查内存缓存：ainvoke 返回时已确认有 interrupt
                if _thread_interrupt_cache.get(thread_id):
                    status = "interrupted"
                else:
                    # next=() 且没检测到中断——可能是竞态窗口：
                    # background task 仍在运行（ainvoke 尚未返回），或
                    # interrupt() 已触发但尚未持久化进 checkpoint
                    task = RUNNING_WORKFLOW_TASKS.get(thread_id)
                    task_still_running = task is not None and not task.done()

                    if task_still_running:
                        # 后台任务仍在执行（如 DataFlowEvalNode 长时间运行），
                        # checkpoint 不会更新直到节点完成，直接返回 running
                        status = "running"
                    else:
                        # 任务已结束，做有限次重试以确认 interrupt 是否刚写入 checkpoint
                        benches = current_values.get("benches", [])
                        has_phase2_data = bool(current_values.get("eval_results"))

                        # 检查是否是无参考答案的评测（仅生成模式）
                        has_reference_answers = False
                        for bench in benches:
                            meta = bench.meta if hasattr(bench, 'meta') else {}
                            eval_results = meta.get("eval_results", {}) if isinstance(meta, dict) else {}
                            for model_name, result in eval_results.items():
                                stats = result.get("stats", {})
                                if stats:  # 如果有评估指标，说明有参考答案
                                    has_reference_answers = True
                                    break
                            if has_reference_answers:
                                break

                        # 如果有 benches 但没有参考答案，这是正常的仅生成模式，不是竞态条件
                        if benches and not has_phase2_data and not has_reference_answers and attempt < max_retries - 1:
                            # 对于仅生成模式，直接返回 completed
                            status = "completed"
                        elif (benches and not has_phase2_data) and attempt < max_retries - 1:
                            delay = retry_delays[attempt]
                            log.info(f"[get_status] Race condition detected (attempt {attempt+1}), retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            status = "completed"
            elif is_interrupted:
                status = "interrupted"
            else:
                status = "running"

            log.debug(f"[get_status] thread_id={thread_id}, status={status}, next={next_nodes}, interrupts={len(interrupts) if interrupts else 0}")

            return {
                "thread_id": thread_id,
                "status": status,
                "next_node": next_nodes,
                "state_values": current_values,
                "interrupts": [{"value": i.value} for i in interrupts] if interrupts else [],
                "eval_progress": get_progress(thread_id),
            }

        return {"thread_id": thread_id, "status": "completed"}

@app.post("/api/workflow/resume/{thread_id}")
async def resume_workflow(thread_id: str, req: ResumeWorkflowRequest):
    req.thread_id = thread_id
    # Apply state updates if provided
    if req.state_updates:
        if "target_model" in req.state_updates and isinstance(req.state_updates["target_model"], dict):
            try:
                tm = req.state_updates["target_model"]
                model_name_or_path = (
                    tm.get("model_name_or_path")
                    or tm.get("path")
                    or tm.get("model_path")
                    or tm.get("hf_model_name_or_path")
                )
                if not model_name_or_path:
                    raise ValueError("target_model missing model_name_or_path/path")

                req.state_updates["target_model"] = ModelConfig(
                    model_name_or_path=str(model_name_or_path),
                    is_api=bool(tm.get("is_api", False)),
                    api_url=tm.get("api_url"),
                    api_key=tm.get("api_key"),
                    api_provider=tm.get("api_provider", "openai_compatible"),
                    temperature=float(tm.get("temperature", 0.0) or 0.0),
                    top_p=float(tm.get("top_p", 1.0) or 1.0),
                    top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
                    repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
                    max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                    seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
                    tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                    max_model_len=tm.get("max_model_len"),
                    gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
                )
            except Exception as e:
                log.error(f"Failed to parse target_model update: {e}")
                del req.state_updates["target_model"]

        # Deserialize nested objects if needed
        if "benches" in req.state_updates and isinstance(req.state_updates["benches"], list):
            # Convert dicts back to BenchInfo objects
            benches_data = req.state_updates["benches"]
            incoming_benches = [
                _coerce_bench_info(b) if isinstance(b, dict) else b 
                for b in benches_data
            ]
            async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
                graph = build_complete_workflow(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": req.thread_id}}
                try:
                    snap = await graph.aget_state(config)
                except Exception:
                    snap = None
                current_values = snap.values if snap and getattr(snap, "values", None) else {}
                current_any = (current_values or {}).get("benches") or []
                current_benches = []
                for b in current_any:
                    try:
                        current_benches.append(_coerce_bench_info(b) if isinstance(b, dict) else b)
                    except Exception:
                        continue
                req.state_updates["benches"] = _merge_benches_preserve_runtime(incoming_benches, current_benches)

        async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
            graph = build_complete_workflow(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": req.thread_id}}

            # 检查工作流是否处于中断状态
            try:
                snap = await graph.aget_state(config)
                is_interrupted = snap and (snap.interrupts or (snap.next and any(node in snap.next for node in ["HumanReviewNode", "PreEvalReviewNode", "MetricReviewNode"])))
            except Exception:
                is_interrupted = False

            if is_interrupted:
                log.warning(f"Workflow {req.thread_id} is interrupted, skipping state updates")
            else:
                log.info(f"Applying state updates for {req.thread_id}: {req.state_updates.keys()}")
                try:
                    await graph.aupdate_state(config, req.state_updates)
                except Exception as e:
                    log.error(f"Failed to update state for {req.thread_id}: {e}")
                    # 如果是"Ambiguous update"错误，跳过更新
                    if "Ambiguous update" in str(e):
                        log.warning("Skipping state update due to ambiguous update error")
                    # 对于其他错误，也跳过更新，继续执行

    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": req.thread_id}}
        try:
            snap = await graph.aget_state(config)
        except Exception:
            raise HTTPException(status_code=404, detail="thread not found")
        next_nodes = snap.next or []
        values = snap.values or {}
        if req.action == "approved" and "PreEvalReviewNode" in next_nodes:
            benches_any = values.get("benches") or []
            missing = []
            invalid = []
            for b in benches_any:
                bench_name = None
                eval_type = None
                if isinstance(b, dict):
                    bench_name = b.get("bench_name")
                    eval_type = b.get("bench_dataflow_eval_type")
                    if not eval_type and isinstance(b.get("meta"), dict):
                        eval_type = b["meta"].get("bench_dataflow_eval_type")
                else:
                    bench_name = getattr(b, "bench_name", None)
                    eval_type = getattr(b, "bench_dataflow_eval_type", None)
                    meta = getattr(b, "meta", None)
                    if not eval_type and isinstance(meta, dict):
                        eval_type = meta.get("bench_dataflow_eval_type")
                if not eval_type:
                    missing.append(str(bench_name or "unknown"))
                elif str(eval_type).strip() not in _VALID_EVAL_TYPES:
                    invalid.append(f"{str(bench_name or 'unknown')}({str(eval_type).strip()})")
            if missing:
                raise HTTPException(status_code=400, detail=f"missing eval_type for benches: {', '.join(missing)}")
            if invalid:
                raise HTTPException(status_code=400, detail=f"invalid eval_type for benches: {', '.join(invalid)}")

    command = Command(resume=req.action)

    _thread_interrupt_cache.pop(req.thread_id, None)
    _launch_graph_task(req.thread_id, None, resume_command=command)
    return {"status": "resuming"}

@app.post("/api/workflow/rerun_execution/{thread_id}")
async def rerun_execution(thread_id: str, req: RerunExecutionRequest):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            snap = await graph.aget_state(config)
        except Exception:
            raise HTTPException(status_code=404, detail="thread not found")

        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="thread not found")

        if req.state_updates:
            updates = dict(req.state_updates)

            if "target_model" in updates and isinstance(updates["target_model"], dict):
                try:
                    tm = updates["target_model"]
                    model_name_or_path = (
                        tm.get("model_name_or_path")
                        or tm.get("path")
                        or tm.get("model_path")
                        or tm.get("hf_model_name_or_path")
                    )
                    if not model_name_or_path:
                        raise ValueError("target_model missing model_name_or_path/path")

                    updates["target_model"] = ModelConfig(
                        model_name_or_path=str(model_name_or_path),
                        is_api=bool(tm.get("is_api", False)),
                        api_url=tm.get("api_url"),
                        api_key=tm.get("api_key"),
                        api_provider=tm.get("api_provider", "openai_compatible"),
                        temperature=float(tm.get("temperature", 0.0) or 0.0),
                        top_p=float(tm.get("top_p", 1.0) or 1.0),
                        top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
                        repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
                        max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                        seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
                        tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                        max_model_len=tm.get("max_model_len"),
                        gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
                    )
                except Exception as e:
                    log.error(f"Failed to parse target_model update: {e}")
                    updates.pop("target_model", None)

            if "benches" in updates and isinstance(updates["benches"], list):
                benches_data = updates["benches"]
                updates["benches"] = [
                    _coerce_bench_info(b) if isinstance(b, dict) else b
                    for b in benches_data
                ]

            log.info(f"Applying rerun state updates for {thread_id}: {list(updates.keys())}")
            await graph.aupdate_state(config, updates)

        snap = await graph.aget_state(config)
        values = snap.values or {}
        benches_any = values.get("benches") or []
        if not isinstance(benches_any, list):
            raise HTTPException(status_code=400, detail="invalid state benches")

        benches_list: List[BenchInfo] = []
        for b in benches_any:
            if isinstance(b, BenchInfo):
                benches_list.append(b)
            elif isinstance(b, dict):
                benches_list.append(_coerce_bench_info(b))

        for b in benches_list:
            if req.bench_name and b.bench_name != req.bench_name:
                continue
            b.eval_status = "pending"
            if b.meta is None:
                b.meta = {}
            if isinstance(b.meta, dict):
                for k in ("eval_result", "eval_detail_path", "eval_error", "eval_abnormality"):
                    b.meta.pop(k, None)

        await graph.aupdate_state(config, {"benches": benches_list, "eval_cursor": 0})

    goto_node = "PreEvalReviewNode" if req.goto_confirm else "DataFlowEvalNode"
    _launch_graph_task(thread_id, None, resume_command=Command(goto=goto_node))
    return {"ok": True, "status": "queued", "goto": goto_node}

@app.post("/api/workflow/manual_start")
async def manual_start(req: ManualStartRequest):
    thread_id = str(uuid.uuid4())
    _set_thread_created_at(thread_id)

    # Build target_models list (multi-model support)
    target_models_list: List[ModelConfig] = []

    # Priority: target_models array > single target_model
    if req.target_models:
        for tm in req.target_models:
            model_name_or_path = (
                tm.get("model_name_or_path")
                or tm.get("path")
                or tm.get("model_path")
                or tm.get("hf_model_name_or_path")
            )
            if not model_name_or_path:
                continue
            cfg = ModelConfig(
                model_name_or_path=str(model_name_or_path),
                is_api=bool(tm.get("is_api", False)),
                api_url=tm.get("api_url"),
                api_key=tm.get("api_key"),
                api_provider=tm.get("api_provider", "openai_compatible"),
                temperature=float(tm.get("temperature", 0.0) or 0.0),
                top_p=float(tm.get("top_p", 1.0) or 1.0),
                top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
                repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
                max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
                tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                max_model_len=tm.get("max_model_len"),
                gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
                api_concurrency=int(tm.get("api_concurrency", 16) or 16),
            )
            target_models_list.append(cfg)

    # Fallback to single target_model for backward compatibility
    if not target_models_list and req.target_model:
        tm = req.target_model
        model_name_or_path = (
            tm.get("model_name_or_path")
            or tm.get("path")
            or tm.get("model_path")
            or tm.get("hf_model_name_or_path")
        )
        if model_name_or_path:
            single_model = ModelConfig(
                model_name_or_path=str(model_name_or_path),
                is_api=bool(tm.get("is_api", False)),
                api_url=tm.get("api_url"),
                api_key=tm.get("api_key"),
                api_provider=tm.get("api_provider", "openai_compatible"),
                temperature=float(tm.get("temperature", 0.0) or 0.0),
                top_p=float(tm.get("top_p", 1.0) or 1.0),
                top_k=int(tm.get("top_k", -1) if tm.get("top_k", -1) is not None else -1),
                repetition_penalty=float(tm.get("repetition_penalty", 1.0) or 1.0),
                max_tokens=int(tm.get("max_tokens", 2048) or 2048),
                seed=(int(tm["seed"]) if tm.get("seed") is not None else None),
                tensor_parallel_size=int(tm.get("tensor_parallel_size", 1) or 1),
                max_model_len=tm.get("max_model_len"),
                gpu_memory_utilization=float(tm.get("gpu_memory_utilization", 0.9) or 0.9),
                api_concurrency=int(tm.get("api_concurrency", 16) or 16),
            )
            target_models_list.append(single_model)

    if not target_models_list:
        raise HTTPException(status_code=400, detail="No valid model configuration provided")

    benches: List[BenchInfo] = []
    for b in req.benches:
        meta = b.meta or {}
        benches.append(
            BenchInfo(
                bench_name=b.bench_name,
                bench_dataflow_eval_type=b.bench_dataflow_eval_type,
                meta=meta,
                dataset_cache=b.dataset_cache,
                download_status="success" if b.dataset_cache else None,
                eval_status="pending",
            )
        )

    first_model_name = target_models_list[0].model_name_or_path

    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # 检查是否有旧的checkpoint状态
        try:
            snap = await graph.aget_state(config)
            # 如果工作流已经结束（next=()）或处于中断状态，删除旧的checkpoint
            if snap and (not snap.next or snap.interrupts):
                log.info(f"Clearing old workflow checkpoint for {thread_id} (next={snap.next}, interrupts={snap.interrupts})")
                await checkpointer.delete(config)
        except Exception:
            # 如果没有checkpoint，继续
            pass

        # 设置初始状态
        await graph.aupdate_state(
            config,
            {
                "user_query": req.user_query,
                "target_model_name": req.target_model_name or first_model_name,
                "target_model": target_models_list[0],
                "target_models": target_models_list,
                "benches": benches,
                "eval_cursor": 0,
            },
        )

    # 对于手动启动，从DatasetStructureNode开始，跳过意图理解和基准搜索
    log.info(f"Manual start for {thread_id}, using goto=DatasetStructureNode")
    _launch_graph_task(thread_id, None, resume_command=Command(goto="DatasetStructureNode"))
    return {"thread_id": thread_id, "status": "started", "model_count": len(target_models_list)}

def _bench_download_sync(bench: Dict[str, Any], *, repo_root: Path, overrides: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    def _pick_best_split(splits: List[str], preferred: str) -> str:
        if not splits:
            return preferred
        if preferred in splits:
            return preferred
        for cand in ("test", "validation", "dev", "val", "train"):
            if cand in splits:
                return cand
        fuzzy = [s for s in splits if "test" in s.lower()]
        if fuzzy:
            return fuzzy[0]
        fuzzy = [s for s in splits if "valid" in s.lower() or "dev" in s.lower()]
        if fuzzy:
            return fuzzy[0]
        return splits[0]

    # Handle both dictionary and BenchInfo objects
    if hasattr(bench, '__dict__'):
        # Convert BenchInfo to dictionary
        bench_dict = bench.__dict__.copy()
    else:
        bench_dict = bench.copy() if isinstance(bench, dict) else {}

    meta = bench_dict.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    bench_dict["meta"] = meta

    if overrides.get("repo_id"):
        hf_meta = meta.get("hf_meta") or {}
        if not isinstance(hf_meta, dict):
            hf_meta = {}
        hf_meta["hf_repo"] = overrides["repo_id"]
        meta["hf_meta"] = hf_meta

    dl_config = meta.get("download_config") or {}
    if not isinstance(dl_config, dict):
        dl_config = {}
    if overrides.get("config"):
        dl_config["config"] = overrides["config"]
    if overrides.get("split"):
        dl_config["split"] = overrides["split"]
    if dl_config:
        meta["download_config"] = dl_config

    hf_repo = None
    if isinstance(meta.get("hf_meta"), dict):
        hf_repo = meta["hf_meta"].get("hf_repo")
    if not hf_repo:
        hf_repo = bench_dict.get("bench_name") or bench_dict.get("name") or ""

    if not dl_config:
        dl_config = {"config": "default", "split": "test"}
        meta["download_config"] = dl_config

    config_name = dl_config.get("config", "default")
    split_name = dl_config.get("split", "test")

    structure = meta.get("structure") or {}
    if isinstance(structure, dict) and structure.get("ok"):
        subsets = structure.get("subsets", [])
        if isinstance(subsets, list):
            available_configs = [s.get("subset") for s in subsets if isinstance(s, dict) and s.get("subset")]
            if available_configs and config_name not in available_configs:
                if "main" in available_configs:
                    config_name = "main"
                else:
                    config_name = available_configs[0]
            matched_subset = next((s for s in subsets if isinstance(s, dict) and s.get("subset") == config_name), None)
            raw_splits = (matched_subset or {}).get("splits", []) if isinstance(matched_subset, dict) else []
            available_splits: List[str] = []
            for sp in raw_splits:
                if isinstance(sp, dict) and sp.get("name"):
                    available_splits.append(str(sp.get("name")))
                elif isinstance(sp, str):
                    available_splits.append(sp)
            split_name = _pick_best_split(available_splits, split_name) if available_splits else split_name

    meta["download_config"] = {
        "config": config_name,
        "split": split_name,
        "reason": "auto-corrected by server",
    }

    cache_root = repo_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    safe_repo = str(hf_repo).replace("/", "__")
    filename = f"{safe_repo}__{config_name}__{split_name}.jsonl"
    output_path = cache_root / filename

    safe_bench = str(bench.get("bench_name") or "").replace("/", "__")
    if safe_bench:
        exact = list(cache_root.glob(f"*__{safe_bench}__{config_name}__{split_name}.jsonl"))
        candidates = exact or list(cache_root.glob(f"*__{safe_bench}__{config_name}__*.jsonl")) or list(cache_root.glob(f"*__{safe_bench}__*.jsonl"))
        candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
        if candidates:
            chosen = max(candidates, key=lambda p: p.stat().st_mtime)
            bench_dict["dataset_cache"] = str(chosen)
            bench_dict["download_status"] = "success"
            meta.pop("download_error", None)
            return bench_dict

    if overrides.get("force") and output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    if output_path.exists() and output_path.stat().st_size > 0:
        bench_dict["dataset_cache"] = str(output_path)
        bench_dict["download_status"] = "success"
        meta.pop("download_error", None)
        return bench_dict

    tool = HFDownloadTool(cache_dir=str(cache_root))
    last_err = ""
    ok = False
    for i in range(max_retries):
        res = tool.download_and_convert(
            repo_id=str(hf_repo),
            config_name=str(config_name),
            split=str(split_name),
            output_path=output_path,
        )
        if res.get("ok"):
            ok = True
            break
        last_err = res.get("error") or ""

    if ok and output_path.exists() and output_path.stat().st_size > 0:
        bench_dict["dataset_cache"] = str(output_path)
        bench_dict["download_status"] = "success"
        meta.pop("download_error", None)
    else:
        bench_dict["download_status"] = "failed"
        meta["download_error"] = last_err or "download failed"
    return bench_dict

def _bench_from_dict(b: Any) -> BenchInfo:
    """从前端传来的 dict 安全构建 BenchInfo，过滤掉 BenchInfo 不认识的字段"""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(BenchInfo)}
    filtered = {k: v for k, v in b.items() if k in valid_fields}
    return BenchInfo(**filtered)

def _bench_to_dict(b: Any) -> Optional[Dict[str, Any]]:
    if b is None:
        return None
    if isinstance(b, dict):
        return b
    if hasattr(b, "__dict__"):
        d = dict(getattr(b, "__dict__", {}) or {})
        return d if isinstance(d, dict) else None
    return None

_BENCHINFO_FIELDS = {f.name for f in fields(BenchInfo)}
_VALID_EVAL_TYPES = {
    "key1_text_score",
    "key2_qa",
    "key2_q_ma",
    "key3_q_choices_a",
    "key3_q_choices_as",
    "key3_q_a_rejected",
    "other",  # 自定义/其他类型
}
_RUNTIME_META_KEYS = {
    "eval_result",
    "eval_detail_path",
    "eval_step3_path",
    "eval_progress",
    "eval_error",
    "eval_abnormality",
    "pred_key",
    "ref_key",
    "artifact_paths",
}

def _coerce_bench_info(value: Any) -> BenchInfo:
    if isinstance(value, BenchInfo):
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid bench payload")
    b = dict(value)
    if not b.get("bench_dataflow_eval_type") and isinstance(b.get("eval_type"), str):
        b["bench_dataflow_eval_type"] = b.get("eval_type")
    if isinstance(b.get("bench_dataflow_eval_type"), str):
        et = b.get("bench_dataflow_eval_type").strip()
        if not et or et == "unknown":
            b["bench_dataflow_eval_type"] = None
    b.pop("eval_type", None)
    filtered = {k: v for k, v in b.items() if k in _BENCHINFO_FIELDS}
    return BenchInfo(**filtered)

def _is_empty_like(v: Any) -> bool:
    return v is None or v == "" or v == {} or v == []

def _merge_benches_preserve_runtime(incoming: List[BenchInfo], current: List[BenchInfo]) -> List[BenchInfo]:
    current_map = {b.bench_name: b for b in (current or []) if isinstance(b, BenchInfo) and b.bench_name}
    merged: List[BenchInfo] = []

    for b in incoming or []:
        if not isinstance(b, BenchInfo):
            merged.append(b)
            continue
        cur = current_map.get(b.bench_name)
        if not cur:
            merged.append(b)
            continue

        if _is_empty_like(getattr(b, "eval_status", None)) and not _is_empty_like(getattr(cur, "eval_status", None)):
            b.eval_status = cur.eval_status
        if _is_empty_like(getattr(b, "dataset_cache", None)) and not _is_empty_like(getattr(cur, "dataset_cache", None)):
            b.dataset_cache = cur.dataset_cache
        if _is_empty_like(getattr(b, "download_status", None)) and not _is_empty_like(getattr(cur, "download_status", None)):
            b.download_status = cur.download_status

        meta_new = b.meta if isinstance(b.meta, dict) else {}
        meta_cur = cur.meta if isinstance(cur.meta, dict) else {}
        for k in _RUNTIME_META_KEYS:
            if _is_empty_like(meta_new.get(k)) and not _is_empty_like(meta_cur.get(k)):
                meta_new[k] = meta_cur.get(k)
        b.meta = meta_new
        merged.append(b)

    return merged

async def _redownload_bench_background(thread_id: str, bench_name: str, overrides: Dict[str, Any]):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        snap = await graph.aget_state(config)
        if not snap or not snap.values:
            return
        values = snap.values
        benches_any = values.get("benches") or []
        if not isinstance(benches_any, list):
            return
        benches = [x for x in (_bench_to_dict(b) for b in benches_any) if isinstance(x, dict)]

        idx = None
        for i, b in enumerate(benches):
            if b.get("bench_name") == bench_name:
                idx = i
                break
        if idx is None:
            return

        bench = benches[idx]
        updated = await asyncio.to_thread(_bench_download_sync, bench, repo_root=REPO_ROOT, overrides=overrides)
        benches[idx] = updated
        await graph.aupdate_state(config, {"benches": [_coerce_bench_info(b) for b in benches]})

@app.post("/api/workflow/redownload/{thread_id}")
async def redownload_bench(thread_id: str, req: RedownloadBenchRequest):
    async with get_checkpointer(DB_PATH, mode="run") as checkpointer:
        graph = build_complete_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snap = await graph.aget_state(config)
        except Exception:
            raise HTTPException(status_code=404, detail="thread not found")

        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="thread not found")

        values = snap.values
        benches_any = values.get("benches") or []
        if not isinstance(benches_any, list):
            raise HTTPException(status_code=400, detail="invalid state benches")
        benches = [x for x in (_bench_to_dict(b) for b in benches_any) if isinstance(x, dict)]

        idx = None
        for i, b in enumerate(benches):
            if b.get("bench_name") == req.bench_name:
                idx = i
                break
        if idx is None:
            raise HTTPException(status_code=404, detail="bench not found")

        bench = benches[idx]

        bench["download_status"] = "pending"
        meta = bench.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        meta.pop("download_error", None)
        bench["meta"] = meta
        benches[idx] = bench
        await graph.aupdate_state(config, {"benches": [_coerce_bench_info(b) for b in benches]})

    overrides = {"repo_id": req.repo_id, "config": req.config, "split": req.split, "force": req.force}
    asyncio.create_task(_redownload_bench_background(thread_id, req.bench_name, overrides))
    return {"ok": True, "status": "queued"}

@app.get("/api/workflow/history", response_model=List[HistoryItem])
async def get_history():
    if not DB_PATH.exists():
        return []
        
    items = []
    thread_meta = _load_thread_meta()
    meta_dirty = False
    try:
        # Optimize: Reuse single connection/checkpointer for all lookups
        async with get_checkpointer(DB_PATH, mode="run") as cp:
            # 1. Get thread_ids using the same connection
            async with cp.conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC LIMIT 50") as cursor:
                rows = await cursor.fetchall()

            # 2. Build graph once
            graph = build_complete_workflow(checkpointer=cp)
            
            for (tid,) in rows:
                cfg = {"configurable": {"thread_id": tid}}
                try:
                    # aget_state uses the passed checkpointer (cp)
                    snap = await graph.aget_state(cfg)
                    if snap and snap.values:
                        q = snap.values.get("user_query", "Unknown Query")
                        # Determine status
                        status = "completed"
                        if snap.next:
                            status = "interrupted" if ("HumanReviewNode" in snap.next or "PreEvalReviewNode" in snap.next or "MetricReviewNode" in snap.next) else "running"
                        # If no next and no error -> completed
                        
                        ts = snap.metadata.get("created_at") if snap.metadata else None
                        meta_item = thread_meta.get(tid) if isinstance(thread_meta.get(tid), dict) else {}
                        created_ts = meta_item.get("created_at")
                        if not created_ts and isinstance(ts, str) and ts.strip():
                            created_ts = ts.strip()
                        if not created_ts:
                            legacy_ts = meta_item.get("updated_at")
                            if isinstance(legacy_ts, str) and legacy_ts.strip():
                                created_ts = legacy_ts.strip()
                        if not created_ts:
                            created_ts = "1970-01-01T00:00:00+00:00"
                        if not meta_item.get("created_at"):
                            next_meta = dict(meta_item)
                            next_meta["created_at"] = created_ts
                            if not next_meta.get("updated_at"):
                                next_meta["updated_at"] = created_ts
                            thread_meta[tid] = next_meta
                            meta_dirty = True
                        date_str = str(created_ts)
                        
                        items.append(HistoryItem(
                            thread_id=tid,
                            updated_at=str(date_str),
                            user_query=str(q),
                            status=status
                        ))
                except Exception:
                    pass
    except Exception as e:
        log.error(f"Error fetching history: {e}")
        return []
    if meta_dirty:
        _write_json_file(THREAD_META_FILE, thread_meta)

    def _parse_dt(v: str):
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:
            return datetime(1970, 1, 1)

    items.sort(key=lambda x: _parse_dt(x.updated_at), reverse=True)
    return items


@app.delete("/api/workflow/history/{thread_id}")
async def delete_history(thread_id: str):
    """Delete a workflow history item by thread_id."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail="History not found")

    try:
        # Delete checkpoints for this thread_id
        async with get_checkpointer(DB_PATH, mode="run") as cp:
            # Check if thread exists first
            async with cp.conn.execute("SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1", (thread_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="History not found")

            # Delete from checkpoints table (the main table used by langgraph)
            await cp.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))

            # Try to delete from other tables if they exist (langgraph may use different schemas)
            for table in ["checkpoint_blobs", "checkpoint_writes", "checkpoints_writes"]:
                try:
                    await cp.conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                except Exception:
                    pass  # Table doesn't exist, ignore

            await cp.conn.commit()

        # Also remove from thread meta
        thread_meta = _load_thread_meta()
        if thread_id in thread_meta:
            del thread_meta[thread_id]
            _write_json_file(THREAD_META_FILE, thread_meta)

        # Remove from running tasks if present
        if thread_id in RUNNING_WORKFLOW_TASKS:
            del RUNNING_WORKFLOW_TASKS[thread_id]

        return {"status": "deleted", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error deleting history {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def get_models():
    models = _load_json_file(MODELS_FILE, default=[])
    return models if isinstance(models, list) else []

@app.post("/api/models")
def add_model(model: Dict[str, Any]):
    models = _load_json_file(MODELS_FILE, default=[])
    if not isinstance(models, list):
        models = []
    models.append(model)
    _write_json_file(MODELS_FILE, models)
    return {"status": "success"}

@app.delete("/api/models/{index}")
def delete_model(index: int):
    models = _load_json_file(MODELS_FILE, default=[])
    if not isinstance(models, list):
        models = []
    if 0 <= index < len(models):
        models.pop(index)
        _write_json_file(MODELS_FILE, models)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Model not found")

@app.put("/api/models/{index}")
def update_model(index: int, model: Dict[str, Any]):
    models = _load_json_file(MODELS_FILE, default=[])
    if not isinstance(models, list):
        models = []
    if 0 <= index < len(models):
        models[index] = model
        _write_json_file(MODELS_FILE, models)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Model not found")

class ModelTestRequest(BaseModel):
    is_api: bool = False
    path: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    api_provider: str = "openai_compatible"

@app.post("/api/models/test")
def test_model(req: ModelTestRequest):
    """Test model connection - either API or local path"""
    if req.is_api:
        # Test API connection
        if not req.api_url or not req.path:
            raise HTTPException(status_code=400, detail="api_url and path (model name) are required for API models")

        base_url = _normalize_anthropic_base_url(req.api_url.strip()) if req.api_provider == "anthropic" else _normalize_openai_base_url(req.api_url.strip())
        payload = {
            "model": req.path.strip(),
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }

        if req.api_provider == "anthropic":
            # Anthropic Messages API
            headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
            if req.api_key and req.api_key.strip():
                headers["x-api-key"] = req.api_key.strip()
            test_url = f"{base_url}/messages"
        else:
            # OpenAI-compatible
            headers = {"Content-Type": "application/json"}
            if req.api_key and req.api_key.strip():
                headers["Authorization"] = f"Bearer {req.api_key.strip()}"
            test_url = f"{base_url}/chat/completions"

        try:
            import httpx
            with httpx.Client(timeout=30.0) as client:
                r = client.post(test_url, headers=headers, json=payload)
                if 200 <= r.status_code < 300:
                    return {"ok": True, "detail": f"API connection successful ({req.api_provider})"}
                try:
                    err_detail = r.json()
                except:
                    err_detail = r.text[:200]
                return {"ok": False, "detail": f"API returned {r.status_code}: {err_detail}"}
        except Exception as e:
            return {"ok": False, "detail": str(e)}
    else:
        # Test local model path
        if not req.path:
            raise HTTPException(status_code=400, detail="path is required for local models")

        raw = req.path.strip()
        resolved = _normalize_model_path_for_host(raw)
        exists_local = Path(resolved).exists()

        if not exists_local:
            # Check if it's a HuggingFace ID (contains "/" but doesn't start with "/" or "/mnt/")
            if "/" in raw and not raw.startswith("/") and not raw.startswith("\\"):
                # Assume it's a valid HF ID, we can't really test without downloading
                return {"ok": True, "detail": f"HuggingFace ID detected: {raw}"}
            raise HTTPException(status_code=400, detail=f"Model path not found: {resolved}")

        return {"ok": True, "detail": f"Local path exists: {resolved}"}

class ModelLoadTestRequest(BaseModel):
    model_path: str
    tensor_parallel_size: int = 1
    max_tokens: int = 32

@app.post("/api/models/test_load")
def test_model_load(req: ModelLoadTestRequest):
    raw = (req.model_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="model_path is required")
    resolved = _normalize_model_path_for_host(raw)
    exists_local = Path(resolved).exists()
    if not exists_local and (":" in raw or raw.startswith("/mnt/")):
        raise HTTPException(status_code=400, detail=f"Model path not found on current host: {resolved}")
    try:
        from dataflow.serving.local_model_llm_serving import LocalModelLLMServing_vllm
        serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path=resolved,
            vllm_tensor_parallel_size=max(1, int(req.tensor_parallel_size or 1)),
            vllm_max_tokens=max(1, int(req.max_tokens or 32)),
            vllm_temperature=0.0,
        )
        serving.start_serving()
        has_tokenizer = hasattr(serving, "tokenizer")
        serving.cleanup()
        if not has_tokenizer:
            raise RuntimeError("serving started but tokenizer is missing")
        return {"ok": True, "detail": f"Model load test passed: {resolved}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model load test failed: {e}")

@app.get("/api/benches/gallery")
def get_bench_gallery():
    return bench_registry.get_all_benches()


@app.delete("/api/benches/gallery/{bench_name}")
def delete_bench_from_gallery(bench_name: str):
    """从 gallery 中删除 benchmark"""
    # 判断是否使用双文件模式
    if BENCH_GALLERY_PUBLIC_PATH.exists():
        success = bench_registry.delete_bench(
            bench_name,
            str(BENCH_GALLERY_PUBLIC_PATH),
            local_mapping_path=str(BENCH_GALLERY_LOCAL_PATH) if BENCH_GALLERY_LOCAL_PATH.exists() else None
        )
    else:
        success = bench_registry.delete_bench(bench_name, str(BENCH_GALLERY_PATH))
    if success:
        return {"status": "success", "message": f"Bench '{bench_name}' deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Bench '{bench_name}' not found or cannot be deleted")


class AppendAttachmentRequest(BaseModel):
    bench_name: str


@app.post("/api/benches/append_attachment")
async def append_attachment_to_bench(
    file: UploadFile = File(...),
    bench_name: str = Form(...),
):
    """追加附件到已有的 benchmark"""
    import aiofiles

    # 查找 bench
    bench = bench_registry.get_bench_by_name(bench_name)
    if not bench:
        raise HTTPException(status_code=404, detail=f"Bench '{bench_name}' not found")

    # 验证文件类型
    allowed_extensions = {".csv", ".jsonl", ".json", ".xlsx", ".xls", ".txt"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    # 安全化文件名
    safe_bench_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in bench_name)
    timestamp = int(datetime.now().timestamp())
    saved_filename = f"{safe_bench_name}_attachment_{timestamp}{file_ext}"
    saved_path = UPLOAD_DIR / saved_filename

    # 保存文件
    async with aiofiles.open(saved_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # 更新 bench 的 artifact_paths
    meta = bench.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    artifact_paths = meta.get("artifact_paths", [])
    if not isinstance(artifact_paths, list):
        artifact_paths = []
    artifact_paths.append(str(saved_path))
    meta["artifact_paths"] = artifact_paths
    bench["meta"] = meta

    # 更新 bench_gallery.json
    try:
        with open(BENCH_GALLERY_PATH, "r", encoding="utf-8") as f:
            file_data = json.load(f)

        if isinstance(file_data, dict) and "benches" in file_data:
            for i, b in enumerate(file_data["benches"]):
                if b.get("bench_name") == bench.get("bench_name"):
                    file_data["benches"][i] = bench
                    break

            with open(BENCH_GALLERY_PATH, "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Failed to update bench_gallery.json: {e}")

    return {
        "status": "success",
        "file_path": str(saved_path),
        "artifact_paths": artifact_paths
    }


@app.get("/api/benches/preview/{bench_name}")
async def preview_bench_file(bench_name: str, limit: int = 10):
    """预览 benchmark 的数据文件内容

    Args:
        bench_name: benchmark 名称
        limit: 返回的行数限制，0 或负数表示返回全部数据
    """
    bench = bench_registry.get_bench_by_name(bench_name)
    if not bench:
        raise HTTPException(status_code=404, detail=f"Bench '{bench_name}' not found")

    # 获取数据集路径
    dataset_cache = bench.get("dataset_cache")
    if not dataset_cache or not Path(dataset_cache).exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    file_path = Path(dataset_cache)
    file_ext = file_path.suffix.lower()

    # limit <= 0 表示返回全部数据
    return_all = limit <= 0

    try:
        if file_ext == ".jsonl":
            # 读取 JSONL 文件
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not return_all and i >= limit:
                        break
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        rows.append(row)
            # 获取所有字段名
            all_keys = set()
            for row in rows:
                all_keys.update(row.keys())
            return {
                "format": "jsonl",
                "columns": sorted(list(all_keys)),
                "rows": rows,
                "total_shown": len(rows),
            }
        elif file_ext == ".json":
            # 读取 JSON 文件
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = data if return_all else data[:limit]
                all_keys = set()
                for row in rows:
                    if isinstance(row, dict):
                        all_keys.update(row.keys())
                return {
                    "format": "json",
                    "columns": sorted(list(all_keys)),
                    "rows": rows,
                    "total_shown": len(rows),
                }
            else:
                return {"format": "json", "columns": list(data.keys()) if isinstance(data, dict) else [], "rows": [data], "total_shown": 1}
        elif file_ext == ".csv":
            # 读取 CSV 文件
            import pandas as pd
            df = pd.read_csv(file_path, nrows=None if return_all else limit)
            return {
                "format": "csv",
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records"),
                "total_shown": len(df),
            }
        elif file_ext in {".xlsx", ".xls"}:
            # 读取 Excel 文件
            import pandas as pd
            df = pd.read_excel(file_path, nrows=limit)
            return {
                "format": "excel",
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records"),
                "total_shown": len(df),
            }
        else:
            # 读取文本文件
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.rstrip("\n") for line in list(f)[:limit]]
            return {
                "format": "text",
                "columns": ["content"],
                "rows": [{"content": line} for line in lines],
                "total_shown": len(lines),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@app.get("/api/benches/analyze_fields/{bench_name}")
async def analyze_bench_fields(bench_name: str):
    """分析 benchmark 数据文件的字段，识别 prompt/target 等关键字段"""
    bench = bench_registry.get_bench_by_name(bench_name)
    if not bench:
        raise HTTPException(status_code=404, detail=f"Bench '{bench_name}' not found")

    dataset_cache = bench.get("dataset_cache")
    if not dataset_cache or not Path(dataset_cache).exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    file_path = Path(dataset_cache)
    file_ext = file_path.suffix.lower()

    # 分析前 50 行来推断字段类型
    sample_rows = []
    try:
        if file_ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    line = line.strip()
                    if line:
                        sample_rows.append(json.loads(line))
        elif file_ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                sample_rows = data[:50]
            else:
                sample_rows = [data]
        elif file_ext == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path, nrows=50)
            sample_rows = df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    if not sample_rows:
        return {"fields": [], "suggestions": {}}

    # 收集所有字段
    all_fields = set()
    for row in sample_rows:
        if isinstance(row, dict):
            all_fields.update(row.keys())

    # 智能推断字段类型
    suggestions = {
        "prompt_fields": [],  # 可能是 prompt/question 的字段
        "target_fields": [],  # 可能是 target/answer 的字段
        "context_fields": [],  # 可能是 context 的字段
        "field_types": {},  # 每个字段的推断类型
    }

    # 用于识别的关键词
    prompt_keywords = ["prompt", "question", "query", "input", "problem", "instruction", "ask"]
    target_keywords = ["answer", "response", "target", "output", "solution", "label", "ground_truth", "gold"]
    context_keywords = ["context", "passage", "document", "article", "background", "reference"]

    for field in all_fields:
        field_lower = field.lower()

        # 推断是否是 prompt 字段
        for kw in prompt_keywords:
            if kw in field_lower:
                suggestions["prompt_fields"].append({"field": field, "reason": f"包含关键词 '{kw}'"})
                break

        # 推断是否是 target 字段
        for kw in target_keywords:
            if kw in field_lower:
                suggestions["target_fields"].append({"field": field, "reason": f"包含关键词 '{kw}'"})
                break

        # 推断是否是 context 字段
        for kw in context_keywords:
            if kw in field_lower:
                suggestions["context_fields"].append({"field": field, "reason": f"包含关键词 '{kw}'"})
                break

        # 分析字段值类型
        sample_values = [row.get(field) for row in sample_rows if isinstance(row, dict) and field in row]
        if sample_values:
            first_val = sample_values[0]
            if isinstance(first_val, str):
                avg_len = sum(len(str(v)) for v in sample_values if v) / max(len([v for v in sample_values if v]), 1)
                suggestions["field_types"][field] = {
                    "type": "string",
                    "avg_length": round(avg_len, 1),
                    "sample": str(first_val)[:200] if first_val else None
                }
            elif isinstance(first_val, list):
                suggestions["field_types"][field] = {
                    "type": "list",
                    "sample": first_val[:3] if first_val else []
                }
            elif isinstance(first_val, dict):
                suggestions["field_types"][field] = {
                    "type": "object",
                    "keys": list(first_val.keys())[:10]
                }
            elif isinstance(first_val, (int, float)):
                suggestions["field_types"][field] = {
                    "type": "number",
                    "sample": first_val
                }
            else:
                suggestions["field_types"][field] = {
                    "type": type(first_val).__name__,
                    "sample": str(first_val)[:100] if first_val else None
                }

    return {
        "fields": sorted(list(all_fields)),
        "suggestions": suggestions,
        "sample_rows": sample_rows[:5],
        "total_rows_analyzed": len(sample_rows)
    }


@app.get("/api/metrics/registry")
def get_metrics_registry():
    """获取所有注册的 Metric 元数据"""
    return get_registered_metrics_meta()


class AddBenchRequest(BaseModel):
    bench_name: str
    type: str  # 如 "language & reasoning", "safety", "code" 等
    description: str
    dataset_url: Optional[str] = None


@app.post("/api/benches/gallery")
def add_bench_to_gallery(req: AddBenchRequest):
    """添加新的 benchmark 到 gallery"""
    # 构建完整的 bench 数据结构
    bench_data = {
        "bench_name": req.bench_name,
        "bench_table_exist": False,  # 用户添加的默认为 False
        "bench_source_url": req.dataset_url or f"https://huggingface.co/datasets/{req.bench_name}",
        "bench_dataflow_eval_type": None,
        "bench_prompt_template": None,
        "bench_keys": [],
        "meta": {
            "bench_name": req.bench_name,
            "source": "user_added",
            "aliases": [req.bench_name],
            "category": "Knowledge & QA",  # Use a valid default category
            "tags": [req.type],
            "description": req.description,
            "created_at": datetime.now(timezone.utc).isoformat(),  # 添加创建时间
            "hf_meta": {
                "bench_name": req.bench_name,
                "hf_repo": req.bench_name,
                "card_text": "",
                "tags": [req.type],
                "exists_on_hf": True
            }
        }
    }

    success = bench_registry.add_bench(bench_data, str(BENCH_GALLERY_PATH))
    if success:
        return {"status": "success", "bench": bench_data}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to add bench. It may already exist.")


# 目录用于存储上传的数据集
UPLOAD_DIR = SERVER_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/api/benches/upload")
async def upload_bench_dataset(
    file: UploadFile = File(...),
    bench_name: str = Form(...),
    eval_type: str = Form(...),
    description: str = Form(default=""),
):
    """上传本地数据集文件并添加到 gallery（使用双文件脱敏模式）"""
    import shutil
    import aiofiles

    # 验证文件类型
    allowed_extensions = {".csv", ".jsonl", ".json", ".xlsx", ".xls", ".txt"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    # 安全化 bench_name
    safe_bench_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in bench_name)
    if not safe_bench_name:
        safe_bench_name = f"uploaded_{int(time.time())}"

    # 保存文件
    saved_filename = f"{safe_bench_name}{file_ext}"
    saved_path = UPLOAD_DIR / saved_filename

    async with aiofiles.open(saved_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # 根据文件类型推断 key_mapping
    default_key_mapping = {
        "input_question_key": "question",
        "input_target_key": "answer",
    }

    # 对于 jsonl 文件，尝试读取第一行推断字段
    if file_ext in {".jsonl", ".json"}:
        try:
            with open(saved_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    import json
                    sample = json.loads(first_line)
                    keys = list(sample.keys()) if isinstance(sample, dict) else []
                    # 智能推断常见字段
                    for k in keys:
                        k_lower = k.lower()
                        if any(q in k_lower for q in ["question", "query", "prompt", "input", "problem"]):
                            default_key_mapping["input_question_key"] = k
                        if any(a in k_lower for a in ["answer", "response", "target", "output", "solution"]):
                            default_key_mapping["input_target_key"] = k
        except Exception:
            pass

    # 当 eval_type 为 "other" 时，根据实际字段自动推断
    if eval_type.strip().lower() == "other":
        has_answer = "input_target_key" in default_key_mapping
        eval_type = "key2_qa" if has_answer else "key1_text_score"

    # 判断是否使用双文件模式
    use_dual_file_mode = BENCH_GALLERY_PUBLIC_PATH.exists()

    if use_dual_file_mode:
        # 双文件模式：分离公共数据和本地映射
        # 公共数据（脱敏）
        public_bench_data = {
            "bench_name": safe_bench_name,
            "bench_table_exist": True,
            "bench_source_url": f"local://uploaded/{safe_bench_name}",  # 脱敏后的标识符
            "bench_dataflow_eval_type": eval_type,
            "bench_prompt_template": None,
            "bench_keys": list(default_key_mapping.keys()),
            "meta": {
                "bench_name": safe_bench_name,
                "source": "local_upload",
                "is_local_upload": True,  # 标识为本地文件
                "aliases": [safe_bench_name, bench_name],
                "category": "Knowledge & QA",
                "tags": [eval_type, "uploaded", "本地上传评测集"],
                "description": description or f"Uploaded from {file.filename}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "key_mapping": default_key_mapping,
                "hf_meta": {
                    "bench_name": safe_bench_name,
                    "hf_repo": "",
                    "card_text": "",
                    "tags": [eval_type, "uploaded"],
                    "exists_on_hf": False
                }
            }
        }

        # 本地映射（包含敏感路径）
        local_mapping = {
            "actual_source_url": f"local://uploads/{saved_filename}",
            "dataset_cache": str(saved_path),
            "uploaded_filename": saved_filename,
            "original_filename": file.filename,
            "upload_time": datetime.now(timezone.utc).isoformat()
        }

        success = bench_registry.add_local_upload_bench(
            public_bench_data,
            str(BENCH_GALLERY_PUBLIC_PATH),
            str(BENCH_GALLERY_LOCAL_PATH),
            local_mapping
        )

        # 返回时使用完整数据（包含本地路径）
        bench_data = dict(public_bench_data)
        bench_data["bench_source_url"] = local_mapping["actual_source_url"]
        bench_data["dataset_cache"] = local_mapping["dataset_cache"]
    else:
        # 旧模式：单一文件（向后兼容）
        bench_data = {
            "bench_name": safe_bench_name,
            "bench_table_exist": True,
            "bench_source_url": f"local://uploads/{saved_filename}",
            "bench_dataflow_eval_type": eval_type,
            "bench_prompt_template": None,
            "bench_keys": list(default_key_mapping.keys()),
            "dataset_cache": str(saved_path),
            "meta": {
                "bench_name": safe_bench_name,
                "source": "local_upload",
                "aliases": [safe_bench_name, bench_name],
                "category": "Knowledge & QA",
                "tags": [eval_type, "uploaded", "本地上传评测集"],
                "description": description or f"Uploaded from {file.filename}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "key_mapping": default_key_mapping,
                "hf_meta": {
                    "bench_name": safe_bench_name,
                    "hf_repo": "",
                    "card_text": "",
                    "tags": [eval_type, "uploaded"],
                    "exists_on_hf": False
                }
            }
        }
        success = bench_registry.add_bench(bench_data, str(BENCH_GALLERY_PATH))

    if success:
        return {
            "status": "success",
            "bench": bench_data,
            "file_path": str(saved_path),
            "rows_estimate": _count_file_rows(saved_path, file_ext)
        }
    else:
        raise HTTPException(status_code=400, detail=f"Failed to add bench. It may already exist.")


def _count_file_rows(file_path: Path, ext: str) -> int:
    """估算文件行数"""
    try:
        if ext in {".jsonl", ".txt", ".csv"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        elif ext in {".json"}:
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                return 1
        elif ext in {".xlsx", ".xls"}:
            import pandas as pd
            df = pd.read_excel(file_path)
            return len(df)
    except Exception:
        pass
    return -1


# --- Eval result download endpoints ---
# NOTE: /csv/ and /xlsx/ routes MUST be declared before the generic /{thread_id}/... route,
#       otherwise FastAPI matches "csv"/"xlsx" as thread_id.

def _find_result_file(bench_name: str, model_name: str, thread_id: str | None = None):
    """查找评测结果文件，返回路径或 None"""
    import glob
    search_patterns = [
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/step_step1.jsonl",
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/*.jsonl",
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/*.json",
    ]
    if thread_id:
        search_patterns.extend([
            f"{REPO_ROOT}/results/{thread_id}/{bench_name}/{model_name}/*.json",
            f"{REPO_ROOT}/results/{thread_id}/{bench_name}/{model_name}/*.jsonl",
        ])
    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


def _read_result_df(filepath: str) -> "pd.DataFrame":
    """读取结果文件为 DataFrame"""
    import pandas as pd
    if filepath.endswith(".jsonl"):
        return pd.read_json(filepath, lines=True)
    elif filepath.endswith(".json"):
        return pd.read_json(filepath)
    else:
        raise ValueError("Unsupported file format")


def _transform_df_for_export(df: "pd.DataFrame") -> "pd.DataFrame":
    """转码 DataFrame：拆分 generated_ans 为 thinking 和 answer 列"""
    import re
    import pandas as pd

    df = df.copy()
    if "generated_ans" in df.columns:
        thinking_list = []
        answer_list = []
        for val in df["generated_ans"]:
            text = str(val) if val else ""
            think_m = re.search(r"<think[^>]*>([\s\S]*?)</think\s*>\s*", text, re.IGNORECASE)
            ans_m = re.search(r"<answer[^>]*>([\s\S]*?)</answer\s*>", text, re.IGNORECASE)
            thinking_list.append(think_m.group(1).strip() if think_m else "")
            answer_list.append(ans_m.group(1).strip() if ans_m else text)
        df["thinking"] = thinking_list
        df["answer"] = answer_list
        # 把 generated_ans 移到最后一列
        cols = [c for c in df.columns if c != "generated_ans"] + ["generated_ans"]
        df = df[cols]
    return df


@app.get("/api/eval/result/csv/{thread_id}/{bench_name}/{model_name}")
async def download_eval_csv(thread_id: str, bench_name: str, model_name: str):
    """下载评测结果为 CSV 格式（转码：拆分 generated_ans 为 thinking + answer）"""
    from fastapi.responses import StreamingResponse
    import pandas as pd
    import io

    filepath = _find_result_file(bench_name, model_name, thread_id)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Result file not found for {bench_name}/{model_name}.")

    try:
        df = _transform_df_for_export(_read_result_df(filepath))
        buf = io.StringIO()
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)
        from urllib.parse import quote
        safe_name = f"{bench_name}_{model_name}_results.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_name)}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to CSV: {str(e)}")


@app.get("/api/eval/result/xlsx/{thread_id}/{bench_name}/{model_name}")
async def download_eval_xlsx(thread_id: str, bench_name: str, model_name: str):
    """下载评测结果为 XLSX 格式（转码：拆分 generated_ans 为 thinking + answer）"""
    from fastapi.responses import Response
    import pandas as pd
    import io

    filepath = _find_result_file(bench_name, model_name, thread_id)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Result file not found for {bench_name}/{model_name}.")

    try:
        df = _transform_df_for_export(_read_result_df(filepath))
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        from urllib.parse import quote
        safe_name = f"{bench_name}_{model_name}_results.xlsx"
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_name)}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to XLSX: {str(e)}")


@app.get("/api/eval/result/{thread_id}/{bench_name}/{model_name}")
async def download_eval_result(thread_id: str, bench_name: str, model_name: str):
    """下载评测结果文件 - 简化版本，直接从已知路径查找"""
    from fastapi.responses import FileResponse
    import glob

    # 根据日志中的路径模式查找结果文件
    # 路径模式：/Users/t/One-Eval/cache/eval_results/{bench_name}_{model_name}_*/step_step1.jsonl

    # 查找可能的文件路径（从项目根目录开始）
    search_patterns = [
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/step_step1.jsonl",
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/*.jsonl",
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/*.json",
        # 备用路径
        f"{REPO_ROOT}/results/{thread_id}/{bench_name}/{model_name}/*.json",
        f"{REPO_ROOT}/results/{thread_id}/{bench_name}/{model_name}/*.jsonl",
    ]

    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            result_file = files[0]
            # 生成友好的文件名
            filename = f"{bench_name}_{model_name}_results.json"
            if result_file.endswith('.jsonl'):
                filename = filename.replace('.json', '.jsonl')

            return FileResponse(
                path=result_file,
                filename=filename,
                media_type="application/json",
            )

    # 如果没找到，返回简单错误
    raise HTTPException(
        status_code=404,
        detail=f"Result file not found for {bench_name}/{model_name}. Check cache/eval_results/ directory."
    )


@app.get("/api/eval/preview/{thread_id}/{bench_name}/{model_name}")
async def preview_eval_result(thread_id: str, bench_name: str, model_name: str, limit: int = 20):
    """预览评测结果（返回前 N 条记录） - 简化版本"""
    import glob
    import pandas as pd

    # 使用与下载API相同的路径查找逻辑
    search_patterns = [
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/step_step1.jsonl",
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/*.jsonl",
        f"{REPO_ROOT}/cache/eval_results/{bench_name}_{model_name}_*/*.json",
        f"{REPO_ROOT}/results/{thread_id}/{bench_name}/{model_name}/*.json",
        f"{REPO_ROOT}/results/{thread_id}/{bench_name}/{model_name}/*.jsonl",
    ]

    detail_path = None
    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            detail_path = files[0]
            break

    if not detail_path or not os.path.exists(detail_path):
        raise HTTPException(
            status_code=404,
            detail=f"Result file not found for {bench_name}/{model_name}. Check cache/eval_results/ directory."
        )

    # 读取文件
    try:
        if detail_path.endswith(".jsonl"):
            df = pd.read_json(detail_path, lines=True)
        elif detail_path.endswith(".json"):
            df = pd.read_json(detail_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        total = len(df)
        preview_df = df.head(limit)
        return {
            "total": total,
            "limit": limit,
            "file_path": detail_path,
            "columns": list(preview_df.columns),
            "rows": preview_df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")


# ============================================================
# LLM-as-Judge 裁判评分 API
# ============================================================

JUDGE_TASKS: Dict[str, asyncio.Task] = {}
JUDGE_TASK_META: Dict[str, Dict[str, Any]] = {}
JUDGE_META_FILE = DATA_DIR / "judge_meta.json"


def _load_judge_meta():
    """从磁盘恢复裁判任务元数据"""
    global JUDGE_TASK_META
    data = _load_json_file(JUDGE_META_FILE, default={})
    if isinstance(data, dict):
        JUDGE_TASK_META = data
        log.info(f"Restored {len(data)} judge task(s) from {JUDGE_META_FILE}")


def _save_judge_meta():
    """持久化裁判任务元数据到磁盘"""
    _write_json_file(JUDGE_META_FILE, JUDGE_TASK_META)


# 启动时恢复
_load_judge_meta()


class JudgeStartRequest(BaseModel):
    bench_name: str
    thread_id: Optional[str] = None             # 关联的会话 ID
    model_names: List[str]                     # 待评分的模型名列表
    model_data_paths: Dict[str, str]          # {model_name: step1_jsonl_path}
    scoring_prompt: str                        # 执行标准 Prompt
    judge_model: Dict[str, Any]                # 裁判模型配置
    concurrency: int = 5
    question_key: str = "question"
    answer_key: str = "generated_ans"
    context_key: Optional[str] = None


@app.post("/api/judge/start")
async def start_judge(req: JudgeStartRequest):
    """启动裁判评分任务（后台执行）"""
    task_id = f"judge_{uuid.uuid4().hex[:12]}"

    # Fallback: 如果用户没填 api_key / api_url，使用系统 agent 配置
    cfg = load_server_config()
    agent_cfg = cfg.get("agent") or {}

    judge_api_key = req.judge_model.get("api_key")
    if not judge_api_key or not str(judge_api_key).strip():
        judge_api_key = agent_cfg.get("api_key") or ""

    judge_api_url = req.judge_model.get("api_url")
    if not judge_api_url or not str(judge_api_url).strip():
        judge_api_url = agent_cfg.get("base_url") or ""

    judge_model_config = ModelConfig(
        model_name_or_path=req.judge_model.get("model_name_or_path") or req.judge_model.get("path", ""),
        is_api=True,
        api_url=judge_api_url,
        api_key=judge_api_key,
        api_provider=req.judge_model.get("api_provider", "openai_compatible"),
        temperature=0.0,
        max_tokens=4096,
    )

    # 计算预估样本总数
    total_samples = 0
    for path in req.model_data_paths.values():
        if os.path.exists(path):
            total_samples += _count_file_rows(Path(path), Path(path).suffix)

    from one_eval.judges.llm_judge import JudgeTaskConfig, LLMJudge
    from one_eval.judges.score_aggregator import ScoreAggregator

    task_config = JudgeTaskConfig(
        task_id=task_id,
        bench_name=req.bench_name,
        scoring_prompt=req.scoring_prompt,
        judge_model=judge_model_config,
        model_names=req.model_names,
        model_data_paths=req.model_data_paths,
        question_key=req.question_key,
        answer_key=req.answer_key,
        context_key=req.context_key,
        concurrency=req.concurrency,
    )

    JUDGE_TASK_META[task_id] = {
        "task_id": task_id,
        "bench_name": req.bench_name,
        "thread_id": req.thread_id,
        "status": "running",
        "total_samples": total_samples,
        "judged": 0,
        "model_names": req.model_names,
        "models_progress": {},
        "started_at": _now_iso(),
    }
    _save_judge_meta()

    async def _run_judge():
        judge = LLMJudge(task_config)
        aggregator = ScoreAggregator()

        try:
            all_results = await judge.judge_all_models(
                progress_cb=lambda p: _update_judge_progress(task_id, p)
            )

            # 聚合统计
            model_stats = {}
            for name, results in all_results.items():
                model_stats[name] = aggregator.aggregate_model(results)
            comparison = aggregator.aggregate_comparison(all_results)

            # 保存结果
            saved_paths = judge.save_results(all_results)
            summary_path = aggregator.save_summary(
                model_stats, comparison, judge._output_dir
            )

            JUDGE_TASK_META[task_id].update({
                "status": "completed",
                "judged": total_samples,
                "detail_path": saved_paths.get("_merged"),
                "summary_path": summary_path,
                "model_detail_paths": {k: v for k, v in saved_paths.items() if k != "_merged"},
                "completed_at": _now_iso(),
            })
            _save_judge_meta()

        except Exception as e:
            log.error(f"Judge task {task_id} failed: {e}", exc_info=True)
            JUDGE_TASK_META[task_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": _now_iso(),
            })
            _save_judge_meta()

    task = asyncio.create_task(_run_judge())
    JUDGE_TASKS[task_id] = task

    return {
        "task_id": task_id,
        "total_samples": total_samples,
        "estimated_calls": total_samples,
        "status": "started",
    }


def _update_judge_progress(task_id: str, progress: Dict[str, Any]):
    """更新裁判任务进度"""
    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        return
    model_name = progress.get("model_name", "")
    status = progress.get("status", "running")

    if status == "done":
        meta["models_progress"][model_name] = {
            "status": "done",
            "judged": progress.get("judged", 0),
            "total": progress.get("total", 0),
            "percent": 100.0,
        }
    elif status == "running":
        meta["models_progress"][model_name] = {
            "status": "running",
            "judged": progress.get("judged", 0),
            "total": progress.get("total", 0),
            "percent": progress.get("percent", 0.0),
        }

    # 计算总进度
    total_judged = 0
    for mp in meta["models_progress"].values():
        total_judged += mp.get("judged", 0)
    meta["judged"] = total_judged
    meta["percent"] = round(total_judged / max(meta["total_samples"], 1) * 100, 1)
    _save_judge_meta()


@app.get("/api/judge/progress/{task_id}")
async def get_judge_progress(task_id: str):
    """查询裁判评分进度"""
    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Judge task not found")
    return meta


@app.get("/api/judge/result/{task_id}")
async def get_judge_result(task_id: str):
    """获取裁判评分结果（摘要 + 统计）"""
    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Judge task not found")

    summary_path = meta.get("summary_path")
    if not summary_path or not os.path.exists(summary_path):
        return {"task_id": task_id, "status": meta.get("status", "unknown"), "detail": "summary not ready"}

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading summary: {str(e)}")


@app.get("/api/judge/detail/{task_id}")
async def get_judge_detail(task_id: str, model_name: Optional[str] = None, limit: int = 50, offset: int = 0):
    """获取裁判评分明细"""
    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Judge task not found")

    # 查找明细文件
    detail_path = None
    if model_name:
        detail_path = meta.get("model_detail_paths", {}).get(model_name)
    else:
        detail_path = meta.get("detail_path")

    if not detail_path or not os.path.exists(detail_path):
        raise HTTPException(status_code=404, detail="Detail file not found")

    try:
        records = []
        with open(detail_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i < offset:
                    continue
                if limit > 0 and len(records) >= limit:
                    break
                records.append(json.loads(line))
        return {
            "total": _count_file_rows(Path(detail_path), Path(detail_path).suffix),
            "shown": len(records),
            "records": records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading detail: {str(e)}")


@app.get("/api/judge/result/{task_id}/csv")
async def download_judge_csv(task_id: str):
    """下载裁判评分结果为 CSV"""
    from fastapi.responses import StreamingResponse
    import pandas as pd
    import io

    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Judge task not found")

    detail_path = meta.get("detail_path")
    if not detail_path or not os.path.exists(detail_path):
        raise HTTPException(status_code=404, detail="Detail file not found")

    try:
        df = pd.read_json(detail_path, lines=True)
        buf = io.StringIO()
        df.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)
        from urllib.parse import quote
        safe_name = f"judge_{task_id}_results.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_name)}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to CSV: {str(e)}")


@app.get("/api/judge/result/{task_id}/xlsx")
async def download_judge_xlsx(task_id: str):
    """下载裁判评分结果为 XLSX"""
    from fastapi.responses import Response
    import pandas as pd
    import io

    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Judge task not found")

    detail_path = meta.get("detail_path")
    if not detail_path or not os.path.exists(detail_path):
        raise HTTPException(status_code=404, detail="Detail file not found")

    try:
        df = pd.read_json(detail_path, lines=True)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        from urllib.parse import quote
        safe_name = f"judge_{task_id}_results.xlsx"
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe_name)}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting to XLSX: {str(e)}")


@app.get("/api/judge/tasks")
async def list_judge_tasks():
    """列出所有裁判评分任务"""
    return list(JUDGE_TASK_META.values())


@app.get("/api/judge/by-thread/{thread_id}")
async def get_judge_tasks_by_thread(thread_id: str):
    """获取指定会话的所有裁判评分任务"""
    tasks = [m for m in JUDGE_TASK_META.values() if m.get("thread_id") == thread_id]
    return tasks


# === 报告缓存 ===
JUDGE_REPORTS: Dict[str, Dict[str, Any]] = {}
JUDGE_REPORTS_FILE = DATA_DIR / "judge_reports.json"


def _load_judge_reports():
    """从磁盘恢复报告缓存"""
    global JUDGE_REPORTS
    data = _load_json_file(JUDGE_REPORTS_FILE, default={})
    if isinstance(data, dict):
        JUDGE_REPORTS = data
        log.info(f"Restored {len(data)} judge report(s) from {JUDGE_REPORTS_FILE}")


def _save_judge_reports():
    """持久化报告缓存到磁盘"""
    _write_json_file(JUDGE_REPORTS_FILE, JUDGE_REPORTS)


# 启动时恢复
_load_judge_reports()


@app.post("/api/judge/report/{task_id}")
async def generate_judge_report(task_id: str):
    """基于裁判评分结果生成分析报告（调用 LLM 分析）"""
    meta = JUDGE_TASK_META.get(task_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Judge task not found")
    if meta.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Judge task not completed yet")

    summary_path = meta.get("summary_path")
    if not summary_path or not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Summary file not found")

    # 读取聚合结果
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # 读取部分明细（取前 20 条用于 case study）
    detail_path = meta.get("detail_path")
    sample_records = []
    if detail_path and os.path.exists(detail_path):
        with open(detail_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                line = line.strip()
                if line:
                    sample_records.append(json.loads(line))

    # 获取 LLM 配置
    cfg = load_server_config()
    agent_cfg = cfg.get("agent") or {}

    # 构建 prompt 让 LLM 分析
    analysis_prompt = _build_report_prompt(summary_data, sample_records, meta)

    # 调用 LLM 生成分析
    try:
        import httpx
        api_url = agent_cfg.get("base_url", "")
        api_key = agent_cfg.get("api_key", "")
        model_name = agent_cfg.get("model", "")
        agent_provider = agent_cfg.get("provider", "openai_compatible")

        if not api_url or not api_key or not model_name:
            raise HTTPException(status_code=500, detail="Agent LLM not configured for report generation")

        system_content = "你是一个专业的 AI 模型评测分析师，擅长基于评测数据进行深度分析并生成结构化报告。"

        if agent_provider == "anthropic":
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            payload = {
                "model": model_name,
                "system": system_content,
                "messages": [{"role": "user", "content": analysis_prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,
            }
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{api_url}/messages", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content_blocks = data.get("content", [])
                analysis_text = "".join(
                    block.get("text", "") for block in content_blocks if block.get("type") == "text"
                )
        else:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": analysis_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{api_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                llm_response = resp.json()
                analysis_text = llm_response["choices"][0]["message"]["content"]

    except Exception as e:
        log.error(f"Report generation failed for {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")

    # 组装报告数据
    report = _assemble_report(summary_data, sample_records, analysis_text, meta)
    JUDGE_REPORTS[task_id] = report
    _save_judge_reports()

    return report


@app.get("/api/judge/report/{task_id}")
async def get_judge_report(task_id: str):
    """获取已生成的分析报告"""
    report = JUDGE_REPORTS.get(task_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not generated yet")
    return report


def _build_report_prompt(summary_data: dict, sample_records: list, meta: dict) -> str:
    """构建报告生成的 prompt"""
    bench_name = meta.get("bench_name", "unknown")
    model_names = meta.get("model_names", [])

    prompt_parts = [
        f"## 评测任务信息\n",
        f"- Bench: {bench_name}",
        f"- 评测模型: {', '.join(model_names)}\n",
        f"## 聚合评分数据\n",
        f"```json\n{json.dumps(summary_data, indent=2, ensure_ascii=False)}\n```\n",
    ]

    if sample_records:
        prompt_parts.append(f"\n## 部分明细样本（前 20 条）\n")
        prompt_parts.append(f"```json\n{json.dumps(sample_records[:10], indent=2, ensure_ascii=False)}\n```\n")

    prompt_parts.append("""
## 分析要求

请基于以上评测数据生成一份结构化分析报告，包含以下部分：

1. **总体评估**：各模型的综合表现对比，整体排名和评价
2. **维度分析**：各评分维度（如有 think_score / body_score / overall_score）的对比分析
3. **问题诊断**：基于 critical_issue 和 other_issues，分析高频问题和共性问题
4. **Case Study**：挑选 2-3 个典型样本（高分/低分），分析评分原因
5. **改进建议**：针对表现较弱的模型，给出具体改进方向

请用中文输出，保持专业客观的语气。
""")
    return "\n".join(prompt_parts)


def _assemble_report(summary_data: dict, sample_records: list, analysis_text: str, meta: dict) -> dict:
    """组装报告数据结构（兼容前端 ReportView）"""
    model_names = meta.get("model_names", [])
    bench_name = meta.get("bench_name", "")

    # 构建 overall 排名
    model_scores = summary_data.get("model_stats", {})
    overall_ranking = []
    for name in model_names:
        stats = model_scores.get(name, {})
        avg = stats.get("avg_scores", {})
        overall_ranking.append({
            "model": name,
            "overall_score": avg.get("overall_score", 0),
            "think_score": avg.get("think_score", 0),
            "body_score": avg.get("body_score", 0),
        })
    overall_ranking.sort(key=lambda x: x["overall_score"], reverse=True)

    # 构建 radar 数据
    radar_dimensions = ["think_score", "body_score", "overall_score"]
    radar_data = {}
    for name in model_names:
        stats = model_scores.get(name, {})
        avg = stats.get("avg_scores", {})
        radar_data[name] = {dim: avg.get(dim, 0) for dim in radar_dimensions}

    # 构建 error distribution
    error_dist = {}
    for name in model_names:
        stats = model_scores.get(name, {})
        issues = {}
        for rec in stats.get("issue_counts", []):
            issues[rec.get("issue", "unknown")] = rec.get("count", 0)
        error_dist[name] = issues

    return {
        "task_id": meta.get("task_id"),
        "bench_name": bench_name,
        "generated_at": _now_iso(),
        "overall": {
            "ranking": overall_ranking,
        },
        "macro": {
            "radar": {
                "dimensions": radar_dimensions,
                "data": radar_data,
            },
        },
        "diagnostic": {
            "error_distribution": error_dist,
        },
        "analyst": {
            "summary": analysis_text,
        },
        "llm_summary": analysis_text,
    }


if __name__ == "__main__":
    import uvicorn
    # Disable uvloop to allow nest_asyncio patching in synchronous nodes
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
