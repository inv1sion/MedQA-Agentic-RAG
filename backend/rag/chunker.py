"""
三级滑动窗口分块器（层次化分块）。

分块配置：
    L1（粗粒度/根级）：size=1200，overlap=240
    L2（中粒度）：     size=600， overlap=120
    L3（叶子/检索级）：size=300， overlap=60

仅 L3 叶子分块向量化写入 Milvus。
L1 和 L2 父块写入 PostgreSQL，供自动合并使用。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..config import get_settings

settings = get_settings()


@dataclass
class Chunk:
    chunk_id: str
    level: int           # 1、2 或 3
    content: str
    chunk_index: int
    doc_id: str
    parent_l2_id: Optional[str] = None   # L3 分块的 L2 父块 ID
    parent_l1_id: Optional[str] = None   # L3/L2 分块的 L1 祖块 ID
    total_children: int = 0              # 该分块拥有的直接子块数量


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """将 *text* 按字符数切分为重叠滑动窗口列表。"""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start += chunk_size - overlap
    return chunks


def build_hierarchical_chunks(text: str, doc_id: str) -> dict[str, list[Chunk]]:
    """
    将 *text* 切分为三级层次结构。

    返回字典，键如下：
        "l1" – L1 Chunk 列表（粗粒度，根级）
        "l2" – L2 Chunk 列表（中粒度，L1 的子级）
        "l3" – L3 Chunk 列表（叶子级，L2 的子级）
    L2/L3 分块上的 parent_l1_id / parent_l2_id 字段已正确填写。
    """
    l1_texts = _sliding_window(text, settings.L1_CHUNK_SIZE, settings.L1_CHUNK_OVERLAP)
    l1_chunks: list[Chunk] = []
    l2_chunks: list[Chunk] = []
    l3_chunks: list[Chunk] = []

    for i1, l1_text in enumerate(l1_texts):
        l1_id = str(uuid.uuid4())
        l2_texts = _sliding_window(l1_text, settings.L2_CHUNK_SIZE, settings.L2_CHUNK_OVERLAP)

        l2_children_ids: list[str] = []
        for i2, l2_text in enumerate(l2_texts):
            l2_id = str(uuid.uuid4())
            l3_texts = _sliding_window(l2_text, settings.L3_CHUNK_SIZE, settings.L3_CHUNK_OVERLAP)

            l3_children_ids: list[str] = []
            for i3, l3_text in enumerate(l3_texts):
                l3_id = str(uuid.uuid4())
                l3_chunks.append(
                    Chunk(
                        chunk_id=l3_id,
                        level=3,
                        content=l3_text,
                        chunk_index=len(l3_chunks),
                        doc_id=doc_id,
                        parent_l2_id=l2_id,
                        parent_l1_id=l1_id,
                    )
                )
                l3_children_ids.append(l3_id)

            l2_chunks.append(
                Chunk(
                    chunk_id=l2_id,
                    level=2,
                    content=l2_text,
                    chunk_index=len(l2_chunks),
                    doc_id=doc_id,
                    parent_l1_id=l1_id,
                    total_children=len(l3_children_ids),
                )
            )
            l2_children_ids.append(l2_id)

        l1_chunks.append(
            Chunk(
                chunk_id=l1_id,
                level=1,
                content=l1_text,
                chunk_index=i1,
                doc_id=doc_id,
                total_children=len(l2_children_ids),
            )
        )

    return {"l1": l1_chunks, "l2": l2_chunks, "l3": l3_chunks}
