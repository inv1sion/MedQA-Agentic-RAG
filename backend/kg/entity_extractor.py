"""
LLM 驱动的医学实体与关系抽取。

从医学文本中抽取结构化三元组并写入 Neo4j 知识图谱。
使用 structured output 确保抽取结果的格式一致性。
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import get_settings
from . import graph_store

logger = logging.getLogger(__name__)
settings = get_settings()

# ── 抽取结果 Schema ───────────────────────────────────────────────────────────


class Entity(BaseModel):
    label: str = Field(
        ...,
        description="实体类型，取值：Disease / Symptom / Drug / BodyPart / Procedure",
    )
    name: str = Field(..., description="实体名称，如 '高血压'、'阿司匹林'")
    aliases: Optional[str] = Field(None, description="别名，逗号分隔")


class Relation(BaseModel):
    from_label: str = Field(..., description="起始实体类型")
    from_name: str = Field(..., description="起始实体名称")
    rel_type: str = Field(
        ...,
        description="关系类型，如 HAS_SYMPTOM / TREATED_BY / AFFECTS / DIAGNOSED_BY / CONTRAINDICATED_WITH",
    )
    to_label: str = Field(..., description="目标实体类型")
    to_name: str = Field(..., description="目标实体名称")


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ── 抽取提示词 ────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """你是一个医学知识图谱实体关系抽取专家。给定一段医学文本，你需要抽取：

1. **实体**：类型限定为 Disease、Symptom、Drug、BodyPart、Procedure
2. **关系**：类型限定为
   - HAS_SYMPTOM（疾病→症状）
   - TREATED_BY（疾病→药物）
   - AFFECTS（疾病→身体部位）
   - DIAGNOSED_BY（疾病→检查手段）
   - CONTRAINDICATED_WITH（药物→药物，禁忌联用）
   - CAUSES（原因→疾病）

规则：
- 实体名称使用标准医学术语，不要口语化
- 同一实体只输出一次
- 关系必须引用已抽取的实体名称
- 无法识别的内容不要强行抽取"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.0,
        max_tokens=2048,
    )


async def extract_entities_and_relations(text: str) -> ExtractionResult:
    """从文本中抽取医学实体和关系。"""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(ExtractionResult)

    result = await structured_llm.ainvoke(
        [
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=f"请从以下医学文本中抽取实体和关系：\n\n{text[:3000]}"),
        ]
    )
    return result


async def extract_and_persist(text: str, doc_id: Optional[str] = None) -> ExtractionResult:
    """抽取实体/关系并写入 Neo4j 图谱。"""
    result = await extract_entities_and_relations(text)

    # 写入实体
    for entity in result.entities:
        props = {}
        if entity.aliases:
            props["aliases"] = entity.aliases
        if doc_id:
            props["source_doc_id"] = doc_id
        await graph_store.upsert_entity(entity.label, entity.name, props)

    # 写入关系
    for rel in result.relations:
        await graph_store.upsert_relation(
            from_label=rel.from_label,
            from_name=rel.from_name,
            rel_type=rel.rel_type,
            to_label=rel.to_label,
            to_name=rel.to_name,
        )

    logger.info(
        "Extracted %d entities, %d relations from doc %s",
        len(result.entities),
        len(result.relations),
        doc_id or "unknown",
    )
    return result
