"""
PostgreSQL 存储的 L1/L2 父块数据库。

支持 Redis 缓存热点父块，减少数据库访问压力。
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..cache.redis_client import redis_client
from ..models.document import ParentChunk

logger = logging.getLogger(__name__)


class DocStore:
    """父块的增删改查工具类 + Redis 缓存。"""

    async def save_chunk(self, chunk: ParentChunk, db: AsyncSession) -> None:
        db.add(chunk)
        # 调用方负责批量提交。

    async def get_chunk(self, chunk_id: str, db: AsyncSession) -> Optional[ParentChunk]:
        # 优先查 Redis 缓存
        cached = await redis_client.get_parent_chunk(chunk_id)
        if cached:
            # 重建轻量对象（读路径避免 SQLAlchemy ORM 开销）
            return _dict_to_chunk(cached)

        result = await db.execute(select(ParentChunk).where(ParentChunk.id == chunk_id))
        chunk = result.scalar_one_or_none()
        if chunk:
            await redis_client.set_parent_chunk(chunk_id, _chunk_to_dict(chunk))
        return chunk

    async def get_chunks_by_doc(self, doc_id: str, db: AsyncSession) -> list[ParentChunk]:
        result = await db.execute(
            select(ParentChunk)
            .where(ParentChunk.doc_id == doc_id)
            .order_by(ParentChunk.level, ParentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def delete_by_doc(self, doc_id: str, db: AsyncSession) -> list[str]:
        """删除某文档的所有父块，返回其 id 列表以便缓存失效。"""
        result = await db.execute(select(ParentChunk).where(ParentChunk.doc_id == doc_id))
        chunks = list(result.scalars().all())
        ids = [c.id for c in chunks]
        for c in chunks:
            await db.delete(c)
        return ids


def _chunk_to_dict(chunk: ParentChunk) -> dict:
    return {
        "id": chunk.id,
        "doc_id": chunk.doc_id,
        "level": chunk.level,
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
        "total_children": chunk.total_children,
        "parent_id": chunk.parent_id,
        "chunk_metadata": chunk.chunk_metadata,
    }


def _dict_to_chunk(d: dict) -> ParentChunk:
    c = ParentChunk()
    for k, v in d.items():
        setattr(c, k, v)
    return c


doc_store = DocStore()
