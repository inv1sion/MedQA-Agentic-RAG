"""
MedQA RAG 评估一键运行脚本
============================
在项目根目录执行：
    python run_eval.py [--n 20] [--k 5] [--out eval_results.json]

脚本会：
1. 从 Milvus 随机采样 N 条 L3 分块作为合成评估查询
   （query = 分块原文，relevant_chunk_id = 该分块 ID）
2. 对比 4 种策略：
     A. 混合检索 + Rerank（默认）
     B. 混合检索，无 Rerank
     C. 纯稠密检索 + Rerank
     D. 纯稠密检索，无 Rerank
3. 计算 Precision@k / Recall@k / MRR / NDCG@k
4. 打印对比表格，并保存 JSON 报告
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,         # 抑制检索器的 INFO 日志，保持输出干净
    format="%(levelname)s: %(message)s",
)

# ── 必须在 import backend 之前确保工作目录是项目根 ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


async def _fetch_sample_chunks(n: int) -> list[dict]:
    """从 Milvus 集合中随机采样 n 条 L3 分块（text + chunk_id）。"""
    from backend.rag.milvus_store import get_collection

    try:
        col = get_collection()
    except Exception as e:
        print(f"[ERROR] 无法连接 Milvus: {e}")
        print("请确保 docker-compose 已启动（docker compose up -d）。")
        sys.exit(1)

    # 先查总数，再随机偏移采样
    total = col.num_entities
    if total == 0:
        print("[ERROR] Milvus 集合为空，请先上传医疗文档。")
        sys.exit(1)

    print(f"[INFO] Milvus 集合中共有 {total} 条 L3 分块，采样 {min(n, total)} 条…")

    # 用 query() 随机偏移采样；若总量小于 n 则全取
    sample_n = min(n, total)
    offset = random.randint(0, max(0, total - sample_n))
    results = col.query(
        expr="chunk_id != ''",
        output_fields=["chunk_id", "doc_id", "text"],
        offset=offset,
        limit=sample_n,
        consistency_level="Strong",
    )
    return results


async def _run(n: int, k: int, out_path: Path) -> None:
    from backend.database import AsyncSessionFactory, init_db
    from backend.eval.evaluator import EvalQuery, StrategyConfig, run_full_evaluation
    from backend.rag.milvus_store import get_collection  # warmup
    from backend.rag.embedder import bm25_manager
    from backend.config import get_settings

    settings = get_settings()

    # 初始化 PostgreSQL 表（幂等）
    try:
        await init_db()
    except Exception as e:
        print(f"[ERROR] PostgreSQL 连接失败: {e}")
        sys.exit(1)

    # 尝试加载 BM25 模型
    bm25_path = Path(settings.BM25_MODEL_PATH)
    if bm25_path.exists():
        try:
            bm25_manager.load(str(bm25_path))
            print(f"[INFO] BM25 模型已加载：{bm25_path}")
        except Exception as e:
            print(f"[WARN] BM25 加载失败（将使用稠密降级）: {e}")
    else:
        print(f"[WARN] 未找到 BM25 模型文件 {bm25_path}，混合检索将自动降级为稠密检索。")

    # 采样分块
    raw_chunks = await _fetch_sample_chunks(n)

    # 构造合成 EvalQuery：query = 分块文本，relevant = [chunk_id]
    eval_queries = [
        EvalQuery(
            query=c["text"],
            relevant_chunk_ids=[c["chunk_id"]],
            doc_ids=[c["doc_id"]],
        )
        for c in raw_chunks
        if c.get("text") and c.get("chunk_id")
    ]

    if not eval_queries:
        print("[ERROR] 未能构造任何评估查询，请检查 Milvus 数据。")
        sys.exit(1)

    print(f"[INFO] 构造了 {len(eval_queries)} 条合成评估查询（top-k={k}）\n")

    strategies = [
        StrategyConfig(name="hybrid+rerank",    top_k=k, candidate_k=20, use_rerank=True,  use_hybrid=True),
        StrategyConfig(name="hybrid+no_rerank", top_k=k, candidate_k=20, use_rerank=False, use_hybrid=True),
        StrategyConfig(name="dense+rerank",     top_k=k, candidate_k=20, use_rerank=True,  use_hybrid=False),
        StrategyConfig(name="dense+no_rerank",  top_k=k, candidate_k=20, use_rerank=False, use_hybrid=False),
    ]

    t0 = time.time()
    async with AsyncSessionFactory() as db:
        reports = await run_full_evaluation(eval_queries, strategies, db, output_path=out_path)
    elapsed = time.time() - t0

    # ── 打印对比表格 ──────────────────────────────────────────────────────────
    col_w = 20
    header = f"{'Strategy':<{col_w}}  {'P@'+str(k):>8}  {'R@'+str(k):>8}  {'MRR':>8}  {'NDCG@'+str(k):>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in reports:
        mode_info = ", ".join(f"{m}×{c}" for m, c in r.retrieval_mode_counts.items())
        print(
            f"{r.strategy:<{col_w}}  "
            f"{r.avg_precision:>8.4f}  "
            f"{r.avg_recall:>8.4f}  "
            f"{r.avg_mrr:>8.4f}  "
            f"{r.avg_ndcg:>8.4f}"
            f"    [{mode_info}]"
        )
    print(sep)
    print(f"\n耗时 {elapsed:.1f}s，JSON 报告已保存至：{out_path}\n")

    # ── Rerank 提升摘要 ───────────────────────────────────────────────────────
    report_map = {r.strategy: r for r in reports}
    for base, ranked in [("hybrid+no_rerank", "hybrid+rerank"), ("dense+no_rerank", "dense+rerank")]:
        if base in report_map and ranked in report_map:
            delta_p = report_map[ranked].avg_precision - report_map[base].avg_precision
            delta_ndcg = report_map[ranked].avg_ndcg - report_map[base].avg_ndcg
            arrow = "↑" if delta_p >= 0 else "↓"
            print(
                f"Rerank 提升（{base.split('+')[0]}）: "
                f"P@{k} {arrow}{abs(delta_p):.4f}  "
                f"NDCG@{k} {arrow}{abs(delta_ndcg):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="MedQA RAG 评估运行器")
    parser.add_argument("--n",   type=int,  default=20,                  help="采样分块数量（默认 20）")
    parser.add_argument("--k",   type=int,  default=5,                   help="检索 top-k（默认 5）")
    parser.add_argument("--out", type=str,  default="eval_results.json", help="JSON 报告输出路径")
    args = parser.parse_args()

    asyncio.run(_run(args.n, args.k, Path(args.out)))


if __name__ == "__main__":
    main()
