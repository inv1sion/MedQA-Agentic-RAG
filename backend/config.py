from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # ─── 大语言模型配置 ──────────────────────────────────────────────────────
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_API_KEY: str = "sk-placeholder"
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096

    # ─── 嵌入向量配置 ──────────────────────────────────────────────────────────
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_API_KEY: str = "sk-placeholder"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_DIM: int = 1024

    # ─── 重排序配置（Qwen3 Rerank）────────────────────────────────────────────
    RERANK_API_KEY: str = "sk-placeholder"
    RERANK_MODEL: str = "gte-rerank"
    RERANK_BASE_URL: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    RERANK_TOP_N: int = 5
    RERANK_CANDIDATE_K: int = 20

    # ─── Milvus 向量库配置 ──────────────────────────────────────────────────
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: str = ""
    MILVUS_COLLECTION: str = "medqa_chunks"

    # ─── PostgreSQL 数据库配置 ────────────────────────────────────────────────
    POSTGRES_URL: str = "postgresql+asyncpg://medqa:medqa123@localhost:5432/medqa"

    # ─── Redis 缓存配置 ───────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SESSION_TTL: int = 3600
    REDIS_DOC_TTL: int = 7200

    # ─── JWT 鉴权配置 ───────────────────────────────────────────────────────
    JWT_SECRET: str = "CHANGE_ME_IN_PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # ─── BM25 稀疏向量配置 ────────────────────────────────────────────────────
    BM25_MODEL_PATH: str = "./bm25_model.pkl"

    # ─── 三级分块配置（L1 粗粒度 / L2 中粒度 / L3 叶子级）────────────────────
    L1_CHUNK_SIZE: int = 1200
    L1_CHUNK_OVERLAP: int = 240
    L2_CHUNK_SIZE: int = 600
    L2_CHUNK_OVERLAP: int = 120
    L3_CHUNK_SIZE: int = 300
    L3_CHUNK_OVERLAP: int = 60

    # ─── RAG 检索配置 ───────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATE_K: int = 20
    RELEVANCE_SCORE_THRESHOLD: float = 0.5
    MERGE_THRESHOLD: float = 0.5

    # ─── 会话摘要记忆配置 ──────────────────────────────────────────────────────
    SUMMARY_THRESHOLD: int = 20
    SUMMARY_WINDOW: int = 10

    # ─── 应用全局配置 ────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:8000,http://127.0.0.1:8000"
    MAX_UPLOAD_SIZE_MB: int = 50
    # ─── Neo4j 图数据库配置 ───────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medqa123"

    # ─── CLIP 多模态配置 ────────────────────────────────────────────────
    CLIP_MODEL_NAME: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    CLIP_DEVICE: str = "cpu"
    IMAGE_UPLOAD_DIR: str = "./uploads/images"
    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def CHECKPOINT_DB_URI(self) -> str:
        """将 asyncpg 连接串转换为 psycopg 格式，供 LangGraph checkpointer 使用。"""
        return self.POSTGRES_URL.replace("+asyncpg", "")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
