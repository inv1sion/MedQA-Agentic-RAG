"""
Milvus 集合管理，存储 L3 叶子分块并支持混合检索。

字段定义（每个分块）：
  chunk_id        – 主键（VARCHAR）
  doc_id          – 所属文档（VARCHAR，可过滤）
  parent_l2_id    – 父级 L2 分块 ID（VARCHAR）
  parent_l1_id    – 祖级 L1 分块 ID（VARCHAR）
  text            – 分块原文（VARCHAR ≤ 2048 字）
  dense_vector    – FLOAT_VECTOR[dim]
  sparse_vector   – SPARSE_FLOAT_VECTOR（BM25）
  chunk_index     – 分块在文档中的序号（INT64）
  metadata        – JSON 元数据（文件名等）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    RRFRanker,
    connections,
    utility,
)

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_collection: Optional[Collection] = None


def _connect() -> None:
    connections.connect(alias="default", uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)


def get_collection() -> Collection:
    global _collection
    if _collection is None:
        _connect()
        _collection = _ensure_collection()
    return _collection


def _ensure_collection() -> Collection:
    name = settings.MILVUS_COLLECTION
    if utility.has_collection(name):
        col = Collection(name)
        col.load()
        return col

    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="parent_l2_id", dtype=DataType.VARCHAR, max_length=64, default_value=""),
        FieldSchema(name="parent_l1_id", dtype=DataType.VARCHAR, max_length=64, default_value=""),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields=fields, description="MEDQA L3 叶子分块 – 混合检索")
    col = Collection(name=name, schema=schema)

    # 稠密向量 HNSW 索引
    col.create_index(
        "dense_vector",
        {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        },
    )
    # BM25 稀疏向量倒排索引
    col.create_index(
        "sparse_vector",
        {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2},
        },
    )
    # 标量索引干速文档级过滤
    col.create_index("doc_id", index_name="doc_id_idx")

    col.load()
    logger.info("Milvus 集合 '%s' 创建并加载完成。", name)
    return col


def insert_chunks(
    chunk_ids: list[str],
    doc_id: str,
    texts: list[str],
    dense_vectors: list[list[float]],
    sparse_vectors: list,  # scipy csr_matrix 稀疏行列表
    parent_l2_ids: list[str],
    parent_l1_ids: list[str],
    chunk_indices: list[int],
    meta: dict[str, Any],
) -> None:
    col = get_collection()
    rows = []
    for i, cid in enumerate(chunk_ids):
        rows.append(
            {
                "chunk_id": cid,
                "doc_id": doc_id,
                "parent_l2_id": parent_l2_ids[i],
                "parent_l1_id": parent_l1_ids[i],
                "text": texts[i][:2048],
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_vectors[i],
                "chunk_index": chunk_indices[i],
                "metadata": meta,
            }
        )
    col.insert(rows)
    col.flush()


def delete_by_doc_id(doc_id: str) -> None:
    col = get_collection()
    col.delete(expr=f'doc_id == "{doc_id}"')
    col.flush()


def hybrid_search(
    dense_query: list[float],
    sparse_query,          # csr_matrix sparse vector
    candidate_k: int = 20,
    top_k: int = 10,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    col = get_collection()
    expr = None
    if doc_ids:
        ids_str = ", ".join(f'"{d}"' for d in doc_ids)
        expr = f"doc_id in [{ids_str}]"

    dense_req = AnnSearchRequest(
        data=[dense_query],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": max(candidate_k * 2, 64)}},
        limit=candidate_k,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_query],
        anns_field="sparse_vector",
        param={"metric_type": "IP", "params": {}},
        limit=candidate_k,
        expr=expr,
    )

    results = col.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=RRFRanker(k=60),
        limit=top_k,
        output_fields=["chunk_id", "doc_id", "parent_l2_id", "parent_l1_id", "text", "chunk_index", "metadata"],
    )

    hits = []
    for hit in results[0]:
        hits.append(
            {
                "chunk_id": hit.entity.get("chunk_id"),
                "doc_id": hit.entity.get("doc_id"),
                "parent_l2_id": hit.entity.get("parent_l2_id", ""),
                "parent_l1_id": hit.entity.get("parent_l1_id", ""),
                "text": hit.entity.get("text", ""),
                "chunk_index": hit.entity.get("chunk_index", 0),
                "metadata": hit.entity.get("metadata", {}),
                "score": hit.score,
                "level": 3,
            }
        )
    return hits


def dense_search(
    dense_query: list[float],
    candidate_k: int = 20,
    top_k: int = 10,
    doc_ids: Optional[list[str]] = None,
) -> list[dict]:
    col = get_collection()
    expr = None
    if doc_ids:
        ids_str = ", ".join(f'"{d}"' for d in doc_ids)
        expr = f"doc_id in [{ids_str}]"

    results = col.search(
        data=[dense_query],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": max(candidate_k * 2, 64)}},
        limit=top_k,
        expr=expr,
        output_fields=["chunk_id", "doc_id", "parent_l2_id", "parent_l1_id", "text", "chunk_index", "metadata"],
    )

    hits = []
    for hit in results[0]:
        hits.append(
            {
                "chunk_id": hit.entity.get("chunk_id"),
                "doc_id": hit.entity.get("doc_id"),
                "parent_l2_id": hit.entity.get("parent_l2_id", ""),
                "parent_l1_id": hit.entity.get("parent_l1_id", ""),
                "text": hit.entity.get("text", ""),
                "chunk_index": hit.entity.get("chunk_index", 0),
                "metadata": hit.entity.get("metadata", {}),
                "score": hit.score,
                "level": 3,
            }
        )
    return hits
