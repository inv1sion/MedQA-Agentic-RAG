"""
基于结构化 LLM 输出的相关性评分器。

使用 Pydantic 结构化输出，对每个检索分块与用户问题的相关性评分（0–1）。

若所有分块的平均分数 < RELEVANCE_SCORE_THRESHOLD，
则认为检索质量较差，建议触发查询重写。
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GradeScore(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="相关性评分，0=完全不相关，1=高度相关",
    )
    reason: str = Field(..., description="评分理由（一句话）")


_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个医疗文档相关性评估专家。\n"
            "给出问题与文档片段的相关性评分（0.0–1.0）：\n"
            "  1.0 – 文档直接回答了问题\n"
            "  0.7 – 文档包含高度相关信息\n"
            "  0.4 – 部分相关\n"
            "  0.1 – 基本不相关\n"
            "以 JSON 格式返回 {{\"score\": float, \"reason\": str}}。",
        ),
        ("human", "问题: {query}\n\n文档片段:\n{document}"),
    ]
)


def _get_grader_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.0,
    ).with_structured_output(GradeScore)


async def grade_document(query: str, document: str) -> GradeScore:
    chain = _GRADE_PROMPT | _get_grader_llm()
    result = await chain.ainvoke({"query": query, "document": document[:1500]})
    return result


async def grade_documents(query: str, documents: list[str]) -> list[GradeScore]:
    tasks = [grade_document(query, doc) for doc in documents]
    return await asyncio.gather(*tasks)


def needs_rewrite(
    grades: list[GradeScore],
    threshold: float = settings.RELEVANCE_SCORE_THRESHOLD,
) -> bool:
    if not grades:
        return True
    avg = sum(g.score for g in grades) / len(grades)
    logger.debug("相关性平均分=%.2f（阈值=%.2f）", avg, threshold)
    return avg < threshold
