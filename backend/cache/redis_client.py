import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


class RedisClient:
    """会话与父文档缓存的高层级 Redis 辅助类。"""

    # ── 会话缓存 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"session:{session_id}"

    async def get_session_messages(self, session_id: str) -> Optional[list]:
        r = await get_redis()
        raw = await r.get(self._session_key(session_id))
        if raw:
            return json.loads(raw)
        return None

    async def set_session_messages(self, session_id: str, messages: list) -> None:
        r = await get_redis()
        await r.setex(
            self._session_key(session_id),
            settings.REDIS_SESSION_TTL,
            json.dumps(messages, ensure_ascii=False),
        )

    async def invalidate_session(self, session_id: str) -> None:
        r = await get_redis()
        await r.delete(self._session_key(session_id))

    # ── 父文档块缓存 ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    async def get_parent_chunk(self, chunk_id: str) -> Optional[dict]:
        r = await get_redis()
        raw = await r.get(self._chunk_key(chunk_id))
        if raw:
            return json.loads(raw)
        return None

    async def set_parent_chunk(self, chunk_id: str, data: dict) -> None:
        r = await get_redis()
        await r.setex(
            self._chunk_key(chunk_id),
            settings.REDIS_DOC_TTL,
            json.dumps(data, ensure_ascii=False),
        )

    async def invalidate_doc_chunks(self, doc_id: str, chunk_ids: list[str]) -> None:
        r = await get_redis()
        if chunk_ids:
            await r.delete(*[self._chunk_key(cid) for cid in chunk_ids])

    # ── 通用辅助方法 ──────────────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        r = await get_redis()
        raw = await r.get(key)
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        r = await get_redis()
        await r.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    async def delete(self, key: str) -> None:
        r = await get_redis()
        await r.delete(key)


redis_client = RedisClient()
