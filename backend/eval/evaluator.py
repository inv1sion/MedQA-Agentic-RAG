"""
RAG 评估框架。

对不同重排策略的检索质量进行量化评估。

指标
----
- Precision@k  – 检索结果中相关文档的比例
- Recall@k     – 相关文档中被检索到的比例
- MRR          – 首个相关结果的平均倒数排名
- NDCG@k       – 归一化折损累积增益
- Rerank 提升  – 重排 vs 原始检索的改善量

独立脚本使用方式
----------------
    python -m backend.eval.evaluator --config eval_config.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..rag.auto_merger import auto_merge
from ..rag.grader import grade_documents
from ..rag.reranker import rerank
from ..rag.retriever import retrieve

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    query: str
    relevant_chunk_ids: list[str]  # 标准答案相关分块 ID 列表
    doc_ids: Optional[list[str]] = None


@dataclass
class StrategyConfig:
    name: str
    top_k: int = 5
    candidate_k: int = 20
    use_rerank: bool = True
    use_hybrid: bool = True


@dataclass
class QueryResult:
    query: str
    strategy: str
    retrieved_ids: list[str]
    reranked_ids: list[str]
    rerank_scores: list[float]
    grade_scores: list[float]
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    retrieval_mode: str = ""


@dataclass
class EvalReport:
    strategy: str
    num_queries: int
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    avg_ndcg: float
    retrieval_mode_counts: dict[str, int] = field(default_factory=dict)
    per_query: list[QueryResult] = field(default_factory=list)


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for cid in top_k if cid in relevant) / len(top_k)


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for cid in top_k if cid in relevant) / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(retrieved, 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(order: list[str]) -> float:
        return sum(
            (1.0 / math.log2(i + 2)) for i, cid in enumerate(order[:k]) if cid in relevant
        )

    ideal = sorted(retrieved[:k], key=lambda cid: cid in relevant, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(retrieved[:k]) / ideal_dcg


async def evaluate_strategy(
    queries: list[EvalQuery],
    strategy: StrategyConfig,
    db: AsyncSession,
) -> EvalReport:
    """对单个策略配置在所有查询上运行评估。"""
    results: list[QueryResult] = []
    mode_counts: dict[str, int] = {}

    for eq in queries:
        # 召回
        chunks, retrieval_mode = await retrieve(
            query=eq.query,
            candidate_k=strategy.candidate_k,
            top_k=strategy.top_k * 3,
            doc_ids=eq.doc_ids,
            use_hybrid=strategy.use_hybrid,
        )
        mode_counts[retrieval_mode] = mode_counts.get(retrieval_mode, 0) + 1

        # 自动合并
        merged = await auto_merge(chunks, db)

        retrieved_ids = [c.get("chunk_id", "") for c in merged]

        # 重排序
        reranked_ids = retrieved_ids
        rerank_scores: list[float] = []
        if strategy.use_rerank and merged:
            reranked = await rerank(eq.query, merged, top_n=strategy.top_k)
            reranked_ids = [c.get("chunk_id", "") for c in reranked]
            rerank_scores = [c.get("rerank_score", 0.0) or 0.0 for c in reranked]
        else:
            reranked_ids = retrieved_ids[: strategy.top_k]

        # 相关性评分
        texts = [c.get("text", "") for c in merged[: strategy.top_k]]
        grades = await grade_documents(eq.query, texts)
        grade_scores = [g.score for g in grades]

        relevant_set = set(eq.relevant_chunk_ids)
        final_ids = reranked_ids if strategy.use_rerank else retrieved_ids[: strategy.top_k]

        qr = QueryResult(
            query=eq.query,
            strategy=strategy.name,
            retrieved_ids=retrieved_ids,
            reranked_ids=reranked_ids,
            rerank_scores=rerank_scores,
            grade_scores=grade_scores,
            precision_at_k=_precision_at_k(final_ids, relevant_set, strategy.top_k),
            recall_at_k=_recall_at_k(final_ids, relevant_set, strategy.top_k),
            mrr=_mrr(final_ids, relevant_set),
            ndcg_at_k=_ndcg_at_k(final_ids, relevant_set, strategy.top_k),
            retrieval_mode=retrieval_mode,
        )
        results.append(qr)

    n = len(results) or 1
    return EvalReport(
        strategy=strategy.name,
        num_queries=n,
        avg_precision=sum(r.precision_at_k for r in results) / n,
        avg_recall=sum(r.recall_at_k for r in results) / n,
        avg_mrr=sum(r.mrr for r in results) / n,
        avg_ndcg=sum(r.ndcg_at_k for r in results) / n,
        retrieval_mode_counts=mode_counts,
        per_query=results,
    )


async def run_full_evaluation(
    queries: list[EvalQuery],
    strategies: list[StrategyConfig],
    db: AsyncSession,
    output_path: Optional[Path] = None,
) -> list[EvalReport]:
    """评估所有策略，并可选将结果保存为 JSON 文件。"""
    reports: list[EvalReport] = []
    for strategy in strategies:
        logger.info("Evaluating strategy: %s", strategy.name)
        report = await evaluate_strategy(queries, strategy, db)
        reports.append(report)
        logger.info(
            "  P@%d=%.3f  R@%d=%.3f  MRR=%.3f  NDCG@%d=%.3f",
            strategy.top_k, report.avg_precision,
            strategy.top_k, report.avg_recall,
            report.avg_mrr,
            strategy.top_k, report.avg_ndcg,
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(r) for r in reports],
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Evaluation results saved to %s", output_path)

    return reports


# ─── FastAPI 路由（内联定义） ───────────────────────────────────────────────────
# 在 main.py 中通过 eval_router 挂载此路由。

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..auth.dependencies import require_admin
from ..database import get_db

eval_router = APIRouter(prefix="/api/eval", tags=["evaluation"])


class EvalQuerySchema(BaseModel):
    query: str
    relevant_chunk_ids: list[str]
    doc_ids: Optional[list[str]] = None


class StrategyConfigSchema(BaseModel):
    name: str
    top_k: int = 5
    candidate_k: int = 20
    use_rerank: bool = True
    use_hybrid: bool = True


class EvalRequest(BaseModel):
    queries: list[EvalQuerySchema]
    strategies: list[StrategyConfigSchema]


@eval_router.post("/run")
async def run_eval(
    payload: EvalRequest,
    _: Any = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Run RAG evaluation (admin only). Returns per-strategy metrics."""
    eq_list = [
        EvalQuery(
            query=q.query,
            relevant_chunk_ids=q.relevant_chunk_ids,
            doc_ids=q.doc_ids,
        )
        for q in payload.queries
    ]
    strat_list = [
        StrategyConfig(
            name=s.name,
            top_k=s.top_k,
            candidate_k=s.candidate_k,
            use_rerank=s.use_rerank,
            use_hybrid=s.use_hybrid,
        )
        for s in payload.strategies
    ]

    reports = await run_full_evaluation(eq_list, strat_list, db)

    return [
        {
            "strategy": r.strategy,
            "num_queries": r.num_queries,
            "avg_precision": round(r.avg_precision, 4),
            "avg_recall": round(r.avg_recall, 4),
            "avg_mrr": round(r.avg_mrr, 4),
            "avg_ndcg": round(r.avg_ndcg, 4),
            "retrieval_mode_counts": r.retrieval_mode_counts,
        }
        for r in reports
    ]
