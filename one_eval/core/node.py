from abc import ABC, abstractmethod
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger(__name__)

class BaseNode(ABC):
    def __init__(self, name: str, tools=None):
        self.name = name
        self.tools = tools or {}

    @abstractmethod
    def run(self, state: NodeState) -> NodeState:
        """核心执行逻辑"""
        pass

    def __call__(self, state: NodeState) -> NodeState:
        """使节点可调用"""
        return self.run(state)

    def log(self, msg: str):
        log.info(f"[{self.name}] {msg}")

class ExampleNode(BaseNode):  # 之后每个不同的node实现放到one_eval/nodes/下，此处只做示范
    def run(self, state: NodeState) -> NodeState:
        self.log("Running example node")
        return state