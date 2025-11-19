import asyncio
from langgraph.graph import START, END
from one_eval.core.state import NodeState
from one_eval.core.graph import GraphBuilder
from one_eval.toolkits.tool_manager import get_tool_manager
from one_eval.nodes.query_understand_node import QueryUnderstandNode
from one_eval.logger import get_logger

log = get_logger("OneEvalWorkflow")


def build_workflow():
    """
    OneEval Workflow。
    """
    tm = get_tool_manager()


    builder = GraphBuilder(state_model=NodeState, entry_point="QueryUnderstandNode")

    # === 注册节点 ===
    node1 = QueryUnderstandNode()          # 变成实例
    builder.add_node(
        name=node1.name,              # "query_understand"
        func=node1.run,                    # 传 run 函数！！！
    )
    builder.add_edge(START, node1.name)
    builder.add_edge(node1.name, END)

    # === 构建图 ===
    graph = builder.build()
    return graph


async def run_demo(user_query: str):
    log.info(f"[workflow] 输入: {user_query}")

    graph = build_workflow()

    # 初始 state
    initial_state = NodeState(user_query=user_query)

    # 运行 workflow（GraphBuilder.compile() 之后内部会按拓扑执行节点）
    final_state = await graph.ainvoke(initial_state)

    log.info(f"[workflow] 最终状态: {final_state}")

    return final_state


if __name__ == "__main__":
    asyncio.run(
        run_demo("我想评估我的模型在文本 reasoning 任务上的表现")
    )
