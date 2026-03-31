"""
Qwen3 / DashScope API 级重排序模块。

接口：POST /api/v1/services/rerank/text-rerank/text-rerank
返回每个文档的 `rerank_score`（0-1，越高越相关）。
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def rerank(
    query: str,
    chunks: list[dict],
    top_n: Optional[int] = None,
) -> list[dict]:
    """
    使用 DashScope Rerank API 对 *chunks* 按 *query* 进行精排。

    每个 chunk 必须包含 "text" 字段。
    返回新列表，按 rerank_score 降序排列，并附带
    `rerank_score` 和 `rerank_index` 字段。

    若 API 调用失败，则回退为原始列表顺序。
    """
    if not chunks:
        return chunks

    top_n = top_n or settings.RERANK_TOP_N
    documents = [c.get("text", "")[:2000] for c in chunks]

    # DashScope 原生接口格式：input + parameters 嵌套结构
    payload = {
        "model": settings.RERANK_MODEL,
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {
            "top_n": min(top_n, len(chunks)),
            "return_documents": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {settings.RERANK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(settings.RERANK_BASE_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("Rerank API 调用失败（%s），保持原始顺序。", e)
        for i, c in enumerate(chunks):
            c.setdefault("rerank_score", None)
        return chunks

    # 原生接口响应格式：{"output": {"results": [...]}}
    results_raw = data.get("output", {}).get("results", [])

    scored: list[dict] = []
    for item in results_raw:
        idx = item["index"]
        score = item.get("relevance_score", 0.0)
        enriched = {**chunks[idx], "rerank_score": round(score, 4), "rerank_index": idx}
        scored.append(enriched)

    # 按 rerank_score 降序排序
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored
