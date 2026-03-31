"""
稠密向量与稀疏向量（BM25）嵌入工具。

稠密 – OpenAI 兼容嵌入接口（阿里云 DashScope text-embedding-v3）
稀疏 – 来自 pymilvus.model 的 BM25EmbeddingFunction；模型持久化到磁盘。
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from typing import Optional

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── Dense Embedding ──────────────────────────────────────────────────────────


async def embed_dense(texts: list[str]) -> list[list[float]]:
    """Call the OpenAI-compatible embedding endpoint and return float vectors."""
    if not texts:
        return []

    url = f"{settings.EMBEDDING_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.EMBEDDING_MODEL,
        "input": texts,
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # 按 index 排序以保持顺序一致
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


async def embed_dense_query(query: str) -> list[float]:
    vecs = await embed_dense([query])
    return vecs[0]


# ─── Sparse BM25 Embedding ────────────────────────────────────────────────────


class BM25Manager:
    """Manages a persistent BM25EmbeddingFunction with thread-safe async access."""

    def __init__(self, model_path: str = settings.BM25_MODEL_PATH):
        self.model_path = model_path
        self._ef = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """若磁盘上存在已保存的模型，则加载。"""
        async with self._lock:
            if self._ef is not None:
                return
            if os.path.exists(self.model_path):
                loop = asyncio.get_event_loop()
                try:
                    self._ef = await loop.run_in_executor(None, self._load_sync)
                    logger.info("BM25 model loaded from %s", self.model_path)
                except Exception as e:
                    logger.warning("Failed to load BM25 model: %s", e)

    def _load_sync(self):
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    async def fit_and_save(self, corpus: list[str]) -> None:
        """在给定语料库上（重新）训练 BM25 并持久化。"""
        if not corpus:
            return
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

        analyzer = build_default_analyzer(language="zh")
        ef = BM25EmbeddingFunction(analyzer)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ef.fit, corpus)

        async with self._lock:
            self._ef = ef
            await loop.run_in_executor(None, self._save_sync, ef)
        logger.info("BM25 model fitted on %d documents and saved.", len(corpus))

    def _save_sync(self, ef) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(ef, f)

    @property
    def is_fitted(self) -> bool:
        return self._ef is not None

    def encode_documents(self, texts: list[str]) -> list[dict]:
        """返回用于入库的稀疏向量（{term_index: weight} 字典列表，pymilvus 2.5 原生支持）。"""
        if self._ef is None:
            raise RuntimeError("尚未训练 BM25 模型，请先上传至少一个文档。")
        mat = self._ef.encode_documents(texts)
        # csr_matrix → [{int: float}]，逐行转换
        coo = mat.tocoo()
        rows: list[dict] = [{} for _ in range(mat.shape[0])]
        for r, c, v in zip(coo.row, coo.col, coo.data):
            rows[int(r)][int(c)] = float(v)
        return rows

    def encode_query(self, query: str) -> Optional[dict]:
        """返回单个查询稀疏向量（{term_index: weight} 字典）。"""
        if self._ef is None:
            return None
        mat = self._ef.encode_queries([query])
        coo = mat.tocoo()
        return {int(c): float(v) for c, v in zip(coo.col, coo.data)}


# 应用级单例对象
bm25_manager = BM25Manager()
