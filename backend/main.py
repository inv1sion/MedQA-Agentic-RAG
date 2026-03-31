"""
MedQA FastAPI 应用入口。
负责初始化数据库、Redis、Milvus、BM25 模型，并沨册各路由。
"""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .cache.redis_client import close_redis, get_redis
from .config import get_settings
from .database import init_db
from .rag.embedder import bm25_manager
from .rag.milvus_store import get_collection

from .api.auth import router as auth_router
from .api.documents import router as doc_router
from .api.chat import router as chat_router
from .api.admin import router as admin_router
from .eval.evaluator import eval_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动阶段 ──────────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting MedQA backend…")

    # 初始化 PostgreSQL 数据库表
    await init_db()
    logger.info("PostgreSQL tables ready.")

    # 初始化 Redis 连接
    try:
        await get_redis()
        logger.info("Redis connection OK.")
    except Exception as e:
        logger.warning("Redis connection failed (non-fatal): %s", e)

    # 初始化 Milvus 向量集合
    try:
        get_collection()
        logger.info("Milvus collection ready.")
    except Exception as e:
        logger.warning("Milvus init failed (non-fatal): %s", e)

    # 若磁盘上存在 BM25 模型则加载
    try:
        await bm25_manager.load()
        if bm25_manager.is_fitted:
            logger.info("BM25 model loaded.")
        else:
            logger.info("No BM25 model found on disk – will fit after first upload.")
    except Exception as e:
        logger.warning("BM25 load failed (non-fatal): %s", e)

    yield

    # ── 应用关闭钩子 ──────────────────────────────────────────────────────────────────
    logger.info("Shutting down MedQA backend…")
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedQA – 中文医疗问答",
        version="1.0.0",
        description="基于 Agentic RAG 的中文医疗智能问答平台。",
        lifespan=lifespan,
    )

    # 配置 CORS 跨域证书
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册 API 路由
    app.include_router(auth_router)
    app.include_router(doc_router)
    app.include_router(chat_router)
    app.include_router(admin_router)
    app.include_router(eval_router)

    # 健康检查接口
    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    # 将 Vue 前端作为静态文件服务（根路径返回 index.html）
    try:
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
    except Exception:
        logger.info("No frontend/ directory found; skipping static file mount.")

    return app


app = create_app()
