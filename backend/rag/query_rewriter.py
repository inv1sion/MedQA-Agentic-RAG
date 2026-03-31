"""
查询重写策略集。

Step-Back  – 将问题抽象为更基础的医学概念。
HyDE       – 生成假设性答案，再以其作为查询。
Router     – 根据问题上下文选择最优策略。
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.0,
    )


# ─── Step-Back 回退提问 ───────────────────────────────────────────────────────────────

_STEP_BACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位医学知识专家。给定一个具体的医疗问题，请将其改写为一个更基础、"
            "更宏观的医学概念性问题，以便从医学知识库中检索到更相关的背景知识。"
            "只返回改写后的问题，不要有任何解释。",
        ),
        ("human", "原始问题: {query}\n\n改写为更基础的问题:"),
    ]
)


async def step_back_rewrite(query: str) -> str:
    chain = _STEP_BACK_PROMPT | _get_llm()
    result = await chain.ainvoke({"query": query})
    rewritten = result.content.strip()
    logger.debug("回退重写: %r → %r", query, rewritten)
    return rewritten


# ─── HyDE 假设性文档嵌入 ────────────────────────────────────────────────────────

_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位医学专家。根据以下问题，生成一段简洁、专业的假设性答案段落（约100字），"
            "该段落将用于在医学知识库中进行相似性检索。请使用正式的医学语言。",
        ),
        ("human", "问题: {query}\n\n假设性答案:"),
    ]
)


async def hyde_rewrite(query: str) -> str:
    chain = _HYDE_PROMPT | _get_llm()
    result = await chain.ainvoke({"query": query})
    hypothesis = result.content.strip()
    logger.debug("HyDE 假设性答案已生成，原始问题: %r", query)
    return hypothesis


# ─── 策略路由器 ─────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个查询路由器。分析医疗问题并选择最适合的重写策略：\n"
            "- step_back: 问题过于具体，需要先检索背景知识（如机理、病因等）\n"
            "- hyde: 问题较明确，可以通过生成假设答案来改善检索\n"
            "- none: 原始查询已足够好，不需要重写\n"
            "只回复策略名称之一：step_back / hyde / none",
        ),
        ("human", "查询: {query}"),
    ]
)


async def route_rewrite(query: str) -> str:
    """返回三种策略之一：'step_back'、'hyde'、'none'。"""
    chain = _ROUTER_PROMPT | _get_llm()
    result = await chain.ainvoke({"query": query})
    strategy = result.content.strip().lower()
    if strategy not in ("step_back", "hyde", "none"):
        strategy = "none"
    logger.debug("查询重写路由：%r → %s", query, strategy)
    return strategy


async def rewrite_query(query: str, strategy: str = "auto") -> tuple[str, str]:
    """
    返回 (rewritten_query, strategy_used)。
    strategy: 'auto' | 'step_back' | 'hyde' | 'none'
    """
    if strategy == "auto":
        strategy = await route_rewrite(query)

    if strategy == "step_back":
        return await step_back_rewrite(query), "step_back"
    elif strategy == "hyde":
        return await hyde_rewrite(query), "hyde"
    else:
        return query, "none"
