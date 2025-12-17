from __future__ import annotations
from typing import Any, Dict, List

import json
from langchain_core.messages import SystemMessage, HumanMessage

from one_eval.core.agent import CustomAgent
from one_eval.core.state import NodeState
from one_eval.logger import get_logger

log = get_logger("HumanInLoopAgent")


class HumanInLoopAgent(CustomAgent):
    """
    Human-in-the-loop Agent：
    根据：
      - 当前节点信息
      - 校验触发信息
      - 人类反馈
      - 各节点说明 & 已执行节点的输入输出
    给出一个 JSON 决策：
        - action: "continue" | "goto_node"
        - target_node: 要跳转的节点名（仅当 action=="goto_node" 时使用）
        - state_update: 要写回 NodeState 的字段
        - approve_validator: 是否把当前 validator 加入白名单
    """

    @property
    def role_name(self) -> str:
        return "HumanInLoopAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "hitl.system"

    @property
    def task_prompt_template_name(self) -> str:
        return "hitl.task"

    async def run(
        self,
        state: NodeState,
        human_input: Any,
        check_result: Any,
        current_node: str,
        allowed_nodes: List[str],
        validator_id: str,
        node_docs: Dict[str, str] | None = None,
        node_io: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        返回格式（理论上）：
        {
          "action": "continue" | "goto_node",
          "target_node": "QueryUnderstandNode" | ... | null,
          "state_update": {...},
          "approve_validator": true/false
        }
        """

        node_docs = node_docs or {}
        node_io = node_io or {}

        # system prompt
        sys_prompt = self.get_prompt(self.system_prompt_template_name)

        def _safe_dump(obj: Any) -> str:
            try:
                return json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                return str(obj)

        # 为 prompt 准备文本
        docs_str = _safe_dump(node_docs)
        io_str = _safe_dump(node_io)
        check_str = _safe_dump(check_result)
        human_str = _safe_dump(human_input)

        # 当前整体摘要（可选，给模型一点全局 context）
        partial_summary = {
            "current_node": getattr(state, "current_node", None),
            "user_query": getattr(state, "user_query", None),
            "benches": getattr(state, "benches", []),
            "agent_results": getattr(state, "agent_results", {}),  # 暂时为空，先保留
        }
        summary_str = _safe_dump(partial_summary)

        task_prompt = self.get_prompt(
            self.task_prompt_template_name,
            current_node=current_node,
            allowed_nodes=", ".join(allowed_nodes) if allowed_nodes else "",
            node_docs=docs_str,
            node_io=io_str,
            check_result=check_str,
            human_input=human_str,
            partial_summary=summary_str,
        )

        llm = self.create_llm(state)

        resp = await llm.call(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task_prompt),
            ],
            bind_post_tools=False,
        )
        log.debug(f"[HumanInLoopAgent] LLM 原始输出: {resp.content}")

        parsed = self.parse_result(resp.content)

        if not isinstance(parsed, dict):
            log.error(
                "[HumanInLoopAgent] LLM 输出不是 dict，使用默认决策: continue + approve"
            )
            return {
                "action": "continue",
                "target_node": None,
                "state_update": {},
                "approve_validator": True,
            }

        action = parsed.get("action") or "continue"
        target_node = parsed.get("target_node")
        state_update = parsed.get("state_update") or {}
        approve_validator = parsed.get("approve_validator")

        if not isinstance(approve_validator, bool):
            approve_validator = True

        # 安全约束：限制跳转目标
        if action == "goto_node":
            if not target_node or target_node not in allowed_nodes:
                log.warning(
                    f"[HumanInLoopAgent] target_node={target_node} 不在允许列表 {allowed_nodes} 中，强制改为 continue"
                )
                action = "continue"
                target_node = None

        result = {
            "action": action,
            "target_node": target_node,
            "state_update": state_update,
            "approve_validator": approve_validator,
        }

        log.info(f"[HumanInLoopAgent] 决策结果: {result}")
        return result
