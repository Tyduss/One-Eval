from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import operator
import time
import copy
import uuid
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


@dataclass
class MainRequest:
    language: str = "zh"
    target: str = ""

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclass
class MainState:
    request: Annotated[MainRequest, lambda x, y: y] = field(default_factory=MainRequest)
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    agent_results: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict)
    temp_data: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        setattr(self, key, value)

@dataclass
class BenchInfo:
    bench_name: str
    bench_table_exist: bool = False
    bench_source_url: str = None
    bench_dataflow_eval_type: str = None        # specified type in bench eval pipeline
    bench_prompt_template: str = None
    bench_keys: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    dataset_cache: Optional[str] = None

    download_status: Optional[str] = None          # "success", "failed", "pending"
    eval_status: Optional[str] = None              # "success", "failed", "pending"


@dataclass
class ModelConfig:
    """Configuration for LLM Serving"""
    model_name_or_path: str
    is_api: bool = False
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    api_provider: str = "openai_compatible"  # "openai_compatible" | "anthropic"

    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = None
    
    # Local model parameters
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9

    # API concurrency
    api_concurrency: int = 16


@dataclass
class NodeState(MainState):
    """Global state container for One-Eval graph execution."""

    # === 基本元信息 ===
    state_id: Annotated[str, lambda x, y: y] = field(default_factory=lambda: str(uuid.uuid4()))
    current_node: Annotated[Optional[str], lambda x, y: y if y else x] = None
    last_updated: Annotated[float, lambda x, y: y] = field(default_factory=time.time)

    # === 用户输入 / 任务上下文 ===
    user_query: Annotated[Optional[str], lambda x, y: y if y else x] = None
    task_domain: Annotated[Optional[str], lambda x, y: y if y else x] = None
    target_model_name: Annotated[Optional[str], lambda x, y: y if y else x] = None
    use_rag: Annotated[bool, operator.or_] = True                             # 是否使用 RAG 进行 benchmark 推荐
    local_count: Annotated[int, lambda x, y: y] = 3
    hf_count: Annotated[int, lambda x, y: y] = 2
    model_type: Annotated[Optional[str], lambda x, y: y if y else x] = None
    target_model: Annotated[Optional[ModelConfig], lambda x, y: y if y else x] = None
    target_models: Annotated[List[ModelConfig], lambda x, y: y] = field(default_factory=list)
    # temp: Dict[str, Any] = field(default_factory=dict)  # 临时存储，用于中间结果
    reference_model: Annotated[Optional[ModelConfig], lambda x, y: y if y else x] = None

    result: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict) # 某个agent的输出结果

    # === 数据与评测基准 ===
    benches: Annotated[List[BenchInfo], lambda x, y: y] = field(default_factory=list)

    eval_cursor: Annotated[int, lambda x, y: y] = 0

    # === 评测规划与结果 ===
    key_plan: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict)   # 输入输出字段映射
    metric_plan: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict)
    eval_results: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict)
    reports: Annotated[Dict[str, Any], lambda x, y: {**x, **y}] = field(default_factory=dict)

    # === LLM交互历史 === #
    llm_history: Annotated[List[Dict[str, Any]], lambda x, y: x + y] = field(default_factory=list)

    # === 人机交互控制 ===
    waiting_for_human: Annotated[bool, operator.or_] = False
    human_feedback: Annotated[Optional[str], lambda x, y: y if y else x] = None
    approved_warning_ids: Annotated[List[str], operator.add] = field(default_factory=list) # validator 白名单
    
    # === 异常与日志 ===
    error_flag: Annotated[bool, operator.or_] = False
    error_msg: Annotated[Optional[str], lambda x, y: y if y else x] = None

    def update(self, **fields: Any) -> None:
        """Update multiple attributes in one call and refresh timestamp."""
        for name, value in fields.items():
            # 仅更新已存在的字段，避免拼写错误产生新属性
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise AttributeError(f"NodeState has no field '{name}'")
        self.last_updated = time.time()

    def checkpoint(self) -> Dict[str, Any]:
        """Return a deep copy of the current state for safe persistence."""
        # copy.deepcopy 确保嵌套结构不会被外部修改影响
        return copy.deepcopy(self.__dict__)

    def resume(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from a snapshot dictionary."""
        for name, value in snapshot.items():
            if hasattr(self, name):
                setattr(self, name, value)
        self.last_updated = time.time()
