"""
MedQA Agent 自定义工具集。

工具列表
--------
medqa_rag_search   – 完整 RAG 流水线：检索 → 自动合并 → 重排 → 评分 → 重写。
hospital_query     – 医院/门诊查询工具（容易扩展的示例实现）。

两个工具均接受一个 asyncio.Queue 用于实时步骤推送。
由于 LangChain 同步工具在线程池中执行，我们使用
`asyncio.get_event_loop().call_soon_threadsafe` + queue 将事件回传到异步 SSE 循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..rag.auto_merger import auto_merge
from ..rag.grader import grade_documents, needs_rewrite
from ..rag.query_rewriter import rewrite_query
from ..rag.reranker import rerank
from ..rag.retriever import retrieve

logger = logging.getLogger(__name__)
settings = get_settings()


def _push_step(queue: Optional[asyncio.Queue], step: str, data: Optional[dict] = None) -> None:
    """线程安全地将步骤事件推送到 SSE 队列。"""
    if queue is None:
        return
    event = {"type": "step", "step": step, "data": data or {}}
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(queue.put_nowait, event)
    except Exception:
        pass  # 不允许队列错误崩溃工具执行


# ─── RAG 检索工具 ──────────────────────────────────────────────────────────────


class RAGInput(BaseModel):
    query: str = Field(..., description="用于检索医学知识库的查询语句")
    top_k: int = Field(default=5, ge=1, le=20, description="最终返回的文档块数量")
    candidate_k: int = Field(default=20, ge=5, le=50, description="召回阶段的候选数量")
    use_rerank: bool = Field(default=True, description="是否使用 Qwen3 Rerank 进行精排")
    use_hybrid: bool = Field(default=True, description="是否使用混合检索（密集+稀疏）")


def _make_rag_tool(
    db: AsyncSession,
    event_queue: Optional[asyncio.Queue] = None,
    doc_ids: Optional[list[str]] = None,
) -> StructuredTool:
    """db/queue/doc_ids 闭包工厂函数。"""

    async def _rag_fn(
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        use_rerank: bool = True,
        use_hybrid: bool = True,
    ) -> str:
        trace: dict[str, Any] = {
            "query": query,
            "retrieval_mode": None,
            "rewrite_strategy": None,
            "rewritten_query": None,
            "sources": [],
            "grade_scores": [],
            "needed_rewrite": False,
        }

        # ── 第一步：检索 ─────────────────────────────────────────────────────────────────
        _push_step(event_queue, "Searching", {"query": query})
        chunks, retrieval_mode = await retrieve(
            query=query,
            candidate_k=candidate_k,
            top_k=top_k * 3,  # 过采样以备重排
            doc_ids=doc_ids,
            use_hybrid=use_hybrid,
        )
        trace["retrieval_mode"] = retrieval_mode

        if not chunks:
            _push_step(event_queue, "NoResults", {})
            return json.dumps({
                "answer": "KNOWLEDGE_BASE_EMPTY: 知识库中未找到与该问题相关的文档。请直接基于通用医学知识回答，无需再次调用本工具。",
                "trace": trace
            }, ensure_ascii=False)

        # -- 第二步：自动合并 -------------------------------------------------
        _push_step(event_queue, "Merging", {"chunks_before": len(chunks)})
        merged = await auto_merge(chunks, db)

        # -- 第三步：重排序 ---------------------------------------------------
        if use_rerank:
            _push_step(event_queue, "Reranking", {"candidates": len(merged)})
            merged = await rerank(query, merged, top_n=top_k)
        else:
            merged = merged[:top_k]

        # -- 第四步：相关性评分 -----------------------------------------------
        _push_step(event_queue, "Grading", {})
        texts = [c.get("text", "") for c in merged]
        grades = await grade_documents(query, texts)
        trace["grade_scores"] = [{"score": g.score, "reason": g.reason} for g in grades]

        # -- 第五步：必要时重写查询 -------------------------------------------
        if needs_rewrite(grades):
            trace["needed_rewrite"] = True
            _push_step(event_queue, "Rewriting", {})
            rewritten, strategy = await rewrite_query(query)
            trace["rewrite_strategy"] = strategy
            trace["rewritten_query"] = rewritten

            new_chunks, new_mode = await retrieve(
                query=rewritten,
                candidate_k=candidate_k,
                top_k=top_k * 3,
                doc_ids=doc_ids,
                use_hybrid=use_hybrid,
            )
            trace["retrieval_mode"] = new_mode
            new_merged = await auto_merge(new_chunks, db)
            if use_rerank:
                merged = await rerank(rewritten, new_merged, top_n=top_k)
            else:
                merged = new_merged[:top_k]

        # -- 第六步：构建来源列表 ---------------------------------------------
        sources = []
        for c in merged:
            sources.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "text": c.get("text", "")[:500],
                    "level": c.get("level", 3),
                    "score": c.get("score"),
                    "rerank_score": c.get("rerank_score"),
                    "metadata": c.get("metadata", {}),
                    "merged_from": c.get("merged_from"),
                }
            )
        trace["sources"] = sources

        # 推送最终来源事件
        _push_step(event_queue, "Done", {"sources": sources})

        # 为 LLM 构建上下文字符串
        context_parts = [f"【片段{i+1}】{c.get('text', '')}" for i, c in enumerate(merged)]
        context = "\n\n".join(context_parts)

        return json.dumps(
            {"context": context, "trace": trace},
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=_rag_fn,
        name="medqa_rag_search",
        description=(
            "从中文医学知识库中检索与问题相关的医学文献片段。"
            "当需要回答任何医疗、疾病、症状、治疗、药物相关问题时必须使用此工具。"
            "返回相关医学文档片段供生成回答使用。"
        ),
        args_schema=RAGInput,
    )


# ─── 医院查询工具 ──────────────────────────────────────────────────────────────


class HospitalInput(BaseModel):
    city: str = Field(..., description="城市名称，如 '北京'、'上海'")
    specialty: str = Field(..., description="科室或专科名称，如 '心内科'、'神经内科'")
    keyword: Optional[str] = Field(None, description="可选的关键词，如医院名称片段")


def _make_hospital_tool(event_queue: Optional[asyncio.Queue] = None) -> StructuredTool:
    async def _hospital_fn(city: str, specialty: str, keyword: Optional[str] = None) -> str:
        """存根：请用真实医院 API 替换此函数。"""
        _push_step(event_queue, "HospitalLookup", {"city": city, "specialty": specialty})

        # 示例存根数据——请用真实 API 替换
        stub_results = [
            {
                "name": f"{city}市第一人民医院",
                "specialty": specialty,
                "address": f"{city}市中心区医院路1号",
                "phone": "010-12345678",
                "rating": 4.5,
            },
            {
                "name": f"{city}协和医院",
                "specialty": specialty,
                "address": f"{city}市东城区协和路2号",
                "phone": "010-87654321",
                "rating": 4.8,
            },
        ]

        if keyword:
            stub_results = [r for r in stub_results if keyword in r["name"]]

        _push_step(event_queue, "Done", {"hospital_count": len(stub_results)})
        return json.dumps({"hospitals": stub_results}, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_hospital_fn,
        name="hospital_query",
        description=(
            "查询指定城市的医院信息（科室、地址、评分等）。"
            "当用户询问去哪里就医、推荐医院或查找特定科室时使用此工具。"
        ),
        args_schema=HospitalInput,
    )


# ─── 对外工厂函数 ──────────────────────────────────────────────────────────────


def build_tools(
    db: AsyncSession,
    event_queue: Optional[asyncio.Queue] = None,
    doc_ids: Optional[list[str]] = None,
) -> list:
    """返回所有工具，并注入当前请求的 db/queue/doc_ids。"""
    return [
        _make_rag_tool(db=db, event_queue=event_queue, doc_ids=doc_ids),
        _make_hospital_tool(event_queue=event_queue),
    ]
