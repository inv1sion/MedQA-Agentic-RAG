"""
会话记忆管理模块，支持自动摘要压缩。

流程：
  1. 从 Redis 或 PostgreSQL 加载会话消息。
  2. 当消息数 > SUMMARY_THRESHOLD 时，对最旧的 SUMMARY_WINDOW 条消息
     生成单条摘要并持久化。
  3. 将摘要注入系统提示前缀，在不膨胀 token 预算的前提下维持长程上下文。
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..cache.redis_client import redis_client
from ..config import get_settings
from ..models.chat import Message, Session

logger = logging.getLogger(__name__)
settings = get_settings()

_SUMMARY_PROMPT = (
    "以下是一段医疗问答对话历史。请用简洁的中文（不超过200字）总结对话的关键信息、"
    "已讨论的症状/诊断/治疗方案，以及任何重要的用户偏好或背景信息。"
    "只输出摘要，不要其他解释。\n\n{history}"
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.0,
        max_tokens=512,
    )


async def _summarise_messages(messages: list[Message]) -> str:
    """调用 LLM 将一批消息压缩为简短摘要。"""
    lines = []
    for m in messages:
        role = "用户" if m.role == "user" else "助手"
        lines.append(f"{role}: {m.content[:500]}")
    history_text = "\n".join(lines)

    llm = _get_llm()
    result = await llm.ainvoke([HumanMessage(content=_SUMMARY_PROMPT.format(history=history_text))])
    return result.content.strip()


async def load_session_messages(session_id: str, db: AsyncSession) -> list[Message]:
    """按创建时间顺序，从 Redis 缓存或 PostgreSQL 加载消息。"""
    # 优先查 Redis 缓存
    cached = await redis_client.get_session_messages(session_id)
    if cached is not None:
        # 重建轻量对象供内存路径使用
        msgs = []
        for d in cached:
            m = Message()
            for k, v in d.items():
                setattr(m, k, v)
            msgs.append(m)
        return msgs

    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    messages = list(result.scalars().all())

    # 预热缓存
    await redis_client.set_session_messages(session_id, [_msg_to_dict(m) for m in messages])
    return messages


async def maybe_summarise(
    session: Session,
    messages: list[Message],
    db: AsyncSession,
) -> Optional[str]:
    """
    若会话消息过多，则对最旧的一批消息进行摘要并持久化。
    返回更新后的摘要字符串（若未触发新摘要则返回原有摘要）。
    """
    threshold = settings.SUMMARY_THRESHOLD
    window = settings.SUMMARY_WINDOW

    user_assistant_msgs = [m for m in messages if m.role in ("user", "assistant")]
    if len(user_assistant_msgs) <= threshold:
        return session.summary

    # 对最旧的一批消息做摘要
    to_summarise = user_assistant_msgs[:window]
    new_summary_text = await _summarise_messages(to_summarise)

    # 与已有摘要合并
    if session.summary:
        combined = f"【历史摘要】{session.summary}\n\n【最新摘要】{new_summary_text}"
        # 二次摘要以保持简洁
        llm = _get_llm()
        result = await llm.ainvoke(
            [HumanMessage(content=f"请将以下两段摘要合并为一段不超过250字的摘要：\n\n{combined}")]
        )
        new_summary_text = result.content.strip()

    session.summary = new_summary_text
    db.add(session)
    await db.commit()

    logger.info("Session %s summary updated (%d chars).", session.id, len(new_summary_text))
    return new_summary_text


def build_lc_messages(
    messages: list[Message],
    session_summary: Optional[str],
    system_prompt: str,
    include_last_n: int = 20,
) -> list:
    """
    将 SQLAlchemy Message 行转换为 LangChain 消息对象。

    - 在头部插入系统提示（包含可选摘要）。
    - 仅保留最近 include_last_n 条用户/助手消息，控制上下文长度。
    """
    # 构建系统消息
    sys_content = system_prompt
    if session_summary:
        sys_content = f"{system_prompt}\n\n【对话摘要】\n{session_summary}"
    lc_messages = [SystemMessage(content=sys_content)]

    # 取最近 N 条用户/助手消息
    history = [m for m in messages if m.role in ("user", "assistant")]
    history = history[-include_last_n:]

    for m in history:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        else:
            lc_messages.append(AIMessage(content=m.content))

    return lc_messages


def _msg_to_dict(m: Message) -> dict:
    return {
        "id": m.id,
        "session_id": m.session_id,
        "role": m.role,
        "content": m.content,
        "msg_metadata": m.msg_metadata,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }
