"""
自动合并：L3 → L2 → L1。

当某个父块的子块被检索到足够数量时，
用 PostgreSQL 中存储的更丰富父块内容替换子块。

合并阈值由 `settings.MERGE_THRESHOLD`（默认 0.5）控制。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from .doc_store import doc_store

logger = logging.getLogger(__name__)
settings = get_settings()


async def auto_merge(
    chunks: list[dict],
    db: AsyncSession,
    merge_threshold: Optional[float] = None,
) -> list[dict]:
    """
    两阶段合并：先 L3→L2，再 L2→L1。

    参数
    ----
    chunks : 来自 Milvus 的 L3 分块（每个包含 parent_l2_id、parent_l1_id）。
    db     : 用于查询父块的异步数据库会话。

    返回
    ----
    合并后的列表——部分 L3 分块可能被其 L2/L1 父块替换。
    """
    threshold = merge_threshold if merge_threshold is not None else settings.MERGE_THRESHOLD
    merged = await _merge_level(chunks, "parent_l2_id", 2, db, threshold)
    merged = await _merge_level(merged, "parent_l1_id", 1, db, threshold)
    return merged


async def _merge_level(
    chunks: list[dict],
    parent_id_key: str,
    parent_level: int,
    db: AsyncSession,
    threshold: float,
) -> list[dict]:
    """当阈值达到时，将分块合并进其父块。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    no_parent: list[dict] = []

    for c in chunks:
        pid = c.get(parent_id_key, "")
        # 仅对有父级 ID 的块进行分组（父级 ID 存在 → 分组，否则保持原样）
        if pid:
            groups[pid].append(c)
        else:
            no_parent.append(c)

    result: list[dict] = list(no_parent)

    for parent_id, children in groups.items():
        parent = await doc_store.get_chunk(parent_id, db)

        if parent is None:
            result.extend(children)
            continue

        total_children = getattr(parent, "total_children", 0)
        if total_children == 0:
            result.extend(children)
            continue

        ratio = len(children) / total_children
        if ratio >= threshold:
            # 覆盖率达标，用父块替换子块集
            merged_chunk = {
                "chunk_id": str(parent.id),
                "doc_id": str(parent.doc_id),
                "text": parent.content,
                "level": parent_level,
                "parent_l2_id": "",
                "parent_l1_id": str(parent.parent_id) if parent.parent_id else "",
                "chunk_index": parent.chunk_index,
                "metadata": parent.chunk_metadata or {},
                "score": max(c.get("score", 0.0) for c in children),
                "rerank_score": max(
                    (c.get("rerank_score") or 0.0) for c in children
                ) or None,
                "merged_from": [c["chunk_id"] for c in children],
            }
            result.append(merged_chunk)
            logger.debug(
                "将 %d 个 L%d 子块合并入父块 %s（覆盖率=%.2f）",
                len(children), parent_level + 1, parent_id, ratio,
            )
        else:
            result.extend(children)

    return result
