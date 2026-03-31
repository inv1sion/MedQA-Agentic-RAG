"""MedQA 的 LangChain OpenAI-tools Agent。

流式架构
--------
Agent 通过 `agent_executor.astream(stream_mode="messages")` 调用。
每次 yield 的 chunk 可能是 AIMessageChunk（token）或带工具调用的 ToolMessage/AIMessage。
两种情况都会被拦截并推送为 SSE 事件。

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

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from .tools import build_tools

logger = logging.getLogger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """你是 MedQA，一位专业的中文医疗问答助手。你具备以下能力：

1. **医学知识检索**：通过知识库工具检索权威医学文献，为用户提供准确的医疗信息。
2. **医院查询**：帮助用户查找推荐医院和科室信息。
3. **专业建议**：基于检索到的医学文献提供专业、客观的医疗建议。

**重要原则**：
- 回答医疗问题前，优先使用 medqa_rag_search 工具检索相关医学知识。
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


def _build_agent(tools: list):
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        verbose=False,
        return_intermediate_steps=True,
    )


async def stream_agent(
    query: str,
    chat_history: list,
    db: AsyncSession,
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

    # 将工具注入 db/queue/doc_ids 的上下文并构建 agent
    tools = build_tools(db=db, event_queue=event_queue, doc_ids=doc_ids)
    agent_executor = _build_agent(tools)

    summary_section = (
        f"【当前对话摘要】\n{session_summary}" if session_summary else ""
    )
    system_prompt = _SYSTEM_PROMPT.format(summary_section=summary_section)

    agent_input = {
        "input": query,
        "chat_history": chat_history,
        "system_prompt": system_prompt,
    }

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
            async for chunk in agent_executor.astream(
                agent_input,
                config={"run_name": "medqa_agent"},
            ):
                # token 逐字输出
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        content = getattr(msg, "content", "")
                        if content and hasattr(msg, "type") and msg.type == "ai":
                            final_answer += content
                            yield {"type": "token", "content": content}

                # 工具调用开始/결束与中间步骤
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        tool_name = getattr(action, "tool", "unknown")
                        yield {"type": "tool_start", "content": tool_name}

                if "steps" in chunk:
                    for step in chunk["steps"]:
                        tool_name = getattr(step.action, "tool", "unknown")
                        tool_out = step.observation or ""

                        # 尝试从 RAG 工具输出中提取来源信息
                        if tool_name == "medqa_rag_search":
                            try:
                                parsed = json.loads(tool_out)
                                if "trace" in parsed:
                                    srcs = parsed["trace"].get("sources", [])
                                    if srcs:
                                        all_sources.extend(srcs)
                            except Exception:
                                pass

                        yield {"type": "tool_end", "content": tool_name}

                # 最终输出
                if "output" in chunk:
                    output = chunk["output"]
                    if output and not final_answer:
                        final_answer = output
                        yield {"type": "token", "content": output}

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
