"""
混合检索器，内置双向降级机制。

策略：
  1. 尝试混合检索（稠密 + BM25 稀疏）配合 RRF 重排。
  2. 若 BM25 尚未训练 或 Milvus 混合检索失败 → 自动降级为稠密检索。
  3. 返回分块列表，并附带检索元数据。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..config import get_settings
from .embedder import bm25_manager, embed_dense_query
from .milvus_store import dense_search, hybrid_search

logger = logging.getLogger(__name__)
settings = get_settings()


async def retrieve(
    query: str,
    candidate_k: int = settings.RETRIEVAL_CANDIDATE_K,
    top_k: int = settings.RETRIEVAL_TOP_K,
    doc_ids: Optional[list[str]] = None,
    use_hybrid: bool = True,
) -> tuple[list[dict], str]:
    """
    返回 (chunks, retrieval_mode)。
    retrieval_mode 取値：'hybrid'（混合） 或 'dense_fallback'（稠密降级）
    """
    loop = asyncio.get_event_loop()

    # 稠密嵌入始终预先计算
    dense_vec = await embed_dense_query(query)

    # 尝试混合检索
    if use_hybrid and bm25_manager.is_fitted:
        try:
            sparse_vec = await loop.run_in_executor(None, bm25_manager.encode_query, query)
            if sparse_vec is None:
                raise ValueError("稀疏向量为空")

            chunks = await loop.run_in_executor(
                None,
                lambda: hybrid_search(dense_vec, sparse_vec, candidate_k, top_k, doc_ids),
            )
            return chunks, "hybrid"
        except Exception as e:
            logger.warning("混合检索失败（%s），回退至稠密检索。", e)

    # 稠密单独降级检索
    chunks = await loop.run_in_executor(
        None,
        lambda: dense_search(dense_vec, candidate_k, top_k, doc_ids),
    )
    return chunks, "dense_fallback"
