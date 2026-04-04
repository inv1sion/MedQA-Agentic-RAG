"""MedQA 的 LangGraph ReAct Agent。

流式架构
--------
Agent 通过 LangGraph 的 `create_react_agent` 构建有状态图，
并使用 `astream_events(version="v2")` 进行流式推送。
拦截 `on_chat_model_stream`、`on_tool_start`、`on_tool_end` 等事件
并转换为 SSE 事件推送给前端。

持久化记忆
----------
通过 LangGraph PostgresSaver checkpointer 将图执行状态持久化到 PostgreSQL。
每个会话的 `session_id` 作为 `thread_id`，图自动累积多轮对话消息，
新请求仅需传入当次用户消息。`state_modifier` 可调用对象负责注入系统提示
并截断过长历史以控制 token 预算。

实时工具步骤可见性
------------------
工具在执行完成前就向 asyncio.Queue 推送步骤事件。
后台协程持续从队列中取出事件并转发给 SSE 生成器，
使前端在工具仍在运行时就能看到「检索 → 合并 → 精排 → 评分」的实时过程。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from .tools import build_tools

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Checkpointer 单例（懒加载）──────────────────────────────────────
_checkpointer: Optional[AsyncPostgresSaver] = None


async def _get_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = AsyncPostgresSaver.from_conn_string(
            settings.CHECKPOINT_DB_URI,
        )
        await _checkpointer.setup()
    return _checkpointer

_SYSTEM_PROMPT = """你是 MedQA，一位专业的中文医疗问答助手。你具备以下能力：

1. **医学知识检索**：通过知识库工具检索权威医学文献，为用户提供准确的医疗信息。
2. **知识图谱推理**：从医学知识图谱中检索疾病、症状、药物等实体的关系网络，辅助推理诊断。
3. **医学图片匹配**：当用户提供医学图片时，通过跨模态匹配找到相关疾病实体。
4. **医院查询**：帮助用户查找推荐医院和科室信息。
5. **专业建议**：基于检索到的医学文献和知识图谱提供专业、客观的医疗建议。

**重要原则**：
- 回答医疗问题前，优先使用 medqa_rag_search 工具检索相关医学知识。
- 涉及疾病-症状-药物关系时，可同时使用 kg_entity_search 工具从知识图谱获取结构化关系。
- 若用户提供了医学图片路径，使用 image_disease_match 工具进行跨模态匹配。
- 若 medqa_rag_search 返回「知识库中未找到相关内容」，**立即停止重试**，直接基于你的医学训练知识给出专业回答，并注明「以下基于通用医学知识」。
- 同一问题**最多调用 medqa_rag_search 一次**，禁止重复调用。
- 不凭空编造具体数据或药品剂量，对数字类信息需注明仅供参考。
- 对于紧急医疗情况，建议立即就医。
- 使用专业但通俗易懂的中文回答。
- 回答结构清晰，必要时使用列表或分段。

{summary_section}"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        streaming=True,
    )


def _build_graph(tools: list, system_prompt: str, checkpointer: AsyncPostgresSaver):
    """构建 LangGraph ReAct Agent 图（带持久化 checkpointer）。"""
    llm = _get_llm()

    def _state_modifier(state: dict) -> list:
        """注入系统提示并截断过长的历史消息以控制 token 预算。"""
        messages = state["messages"]
        if len(messages) > 20:
            messages = messages[-20:]
        return [SystemMessage(content=system_prompt)] + messages

    return create_react_agent(
        llm,
        tools,
        state_modifier=_state_modifier,
        checkpointer=checkpointer,
    )


async def stream_agent(
    query: str,
    chat_history: list,
    db: AsyncSession,
    session_id: Optional[str] = None,
    session_summary: Optional[str] = None,
    doc_ids: Optional[list[str]] = None,
    top_k: int = 5,
    candidate_k: int = 20,
    use_rerank: bool = True,
    use_hybrid: bool = True,
) -> AsyncGenerator[dict, None]:
    """
    异步生成器，逐个 yield SSE 事件字典：

      {"type": "token",      "content": "..."}
      {"type": "step",       "step": "Searching", "data": {...}}
      {"type": "tool_start", "content": "tool_name"}
      {"type": "tool_end",   "content": "tool_name"}
      {"type": "sources",    "data": [...]}
      {"type": "done",       "data": {"final_answer": "..."}}
      {"type": "error",      "content": "..."}
    """
    event_queue: asyncio.Queue = asyncio.Queue()
    all_sources: list = []

    # 将工具注入 db/queue/doc_ids 的上下文并构建 LangGraph 图
    tools = build_tools(db=db, event_queue=event_queue, doc_ids=doc_ids)

    summary_section = (
        f"【当前对话摘要】\n{session_summary}" if session_summary else ""
    )
    system_prompt = _SYSTEM_PROMPT.format(summary_section=summary_section)

    checkpointer = await _get_checkpointer()
    graph = _build_graph(tools, system_prompt, checkpointer)

    # 根据是否已有 checkpoint 决定输入：
    #   - 已有 checkpoint → 仅传新消息（图状态自动累积历史）
    #   - 无 checkpoint → 用现有 chat_history 播种
    run_config: dict = {"run_name": "medqa_agent", "recursion_limit": 10}
    if session_id:
        run_config["configurable"] = {"thread_id": session_id}
        state = await graph.aget_state(run_config)
        if state and state.values:
            input_messages = [HumanMessage(content=query)]
        else:
            input_messages = list(chat_history) + [HumanMessage(content=query)]
    else:
        input_messages = list(chat_history) + [HumanMessage(content=query)]

    # ── 后台队列消费协程 ──────────────────────────────────────────────────────────
    # 队列消费协程与 agent 流并发运行。
    # 使用哨兵值来停止它。
    _SENTINEL = object()

    async def drain_queue() -> AsyncGenerator[dict, None]:
        while True:
            item = await event_queue.get()
            if item is _SENTINEL:
                break
            yield item

    # ── 流式输出 Agent 结果 ───────────────────────────────────────────────────
    final_answer = ""

    async def _stream_agent_events() -> AsyncGenerator[dict, None]:
        nonlocal final_answer, all_sources
        try:
            async for event in graph.astream_events(
                {"messages": input_messages},
                version="v2",
                config=run_config,
            ):
                kind = event["event"]

                # token 逐字输出
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        final_answer += chunk.content
                        yield {"type": "token", "content": chunk.content}

                # 工具调用开始
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield {"type": "tool_start", "content": tool_name}

                # 工具调用结束
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    tool_output = event["data"].get("output", "")

                    # 尝试从 RAG 工具输出中提取来源信息
                    if tool_name == "medqa_rag_search":
                        try:
                            parsed = (
                                json.loads(tool_output)
                                if isinstance(tool_output, str)
                                else tool_output
                            )
                            if isinstance(parsed, dict) and "trace" in parsed:
                                srcs = parsed["trace"].get("sources", [])
                                if srcs:
                                    all_sources.extend(srcs)
                        except Exception:
                            pass

                    yield {"type": "tool_end", "content": tool_name}

        except asyncio.CancelledError:
            yield {"type": "error", "content": "生成已被用户中止"}
        except Exception as e:
            logger.exception("Agent stream error")
            yield {"type": "error", "content": str(e)}
        finally:
            # 通知队列消费协程停止
            await event_queue.put(_SENTINEL)

    # ── 通过共享输出队列合并两路流 ────────────────────────────────────────────
    out_queue: asyncio.Queue = asyncio.Queue()

    async def _pipe_agent():
        async for ev in _stream_agent_events():
            await out_queue.put(ev)
        await out_queue.put({"type": "__agent_done__"})

    async def _pipe_queue():
        async for ev in drain_queue():
            await out_queue.put(ev)

    pipe_a = asyncio.create_task(_pipe_agent())
    pipe_q = asyncio.create_task(_pipe_queue())

    agent_finished = False

    try:
        while not agent_finished:
            try:
                ev = await asyncio.wait_for(out_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if ev.get("type") == "__agent_done__":
                agent_finished = True
                # 清空队列中残留的工具步骤事件
                while not event_queue.empty():
                    try:
                        remaining = event_queue.get_nowait()
                        if remaining is not _SENTINEL:
                            yield remaining
                    except asyncio.QueueEmpty:
                        break
                break
            else:
                yield ev
    finally:
        pipe_a.cancel()
        pipe_q.cancel()

    # 推送最终来源列表和完成事件
    if all_sources:
        yield {"type": "sources", "data": all_sources}

    yield {"type": "done", "data": {"final_answer": final_answer}}
