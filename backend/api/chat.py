"""Chat routes: sessions, history, and streaming SSE endpoint.

SSE event types (JSON-encoded in `data:` field):
  token       – incremental LLM output token
  step        – RAG step trace (Searching, Merging, Reranking, Grading, Rewriting)
  tool_start  – agent began calling a tool
  tool_end    – agent received tool result
  sources     – final list of retrieved/reranked sources
  done        – generation complete, includes final_answer
  error       – error message
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent.agent import stream_agent
from ..agent.memory import build_lc_messages, load_session_messages, maybe_summarise
from ..auth.dependencies import get_current_user
from ..cache.redis_client import redis_client
from ..database import get_db
from ..models.chat import Message, Session
from ..models.user import User
from ..schemas.chat import (
    ChatRequest,
    MessageResponse,
    SessionCreate,
    SessionResponse,
    SessionWithMessages,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_SYSTEM_PROMPT_BASE = (
    "你是 MedQA，一位专业的中文医疗问答助手。"
    "回答医疗问题前必须先使用知识库检索工具获取权威医学文献，不得凭空编造信息。"
    "对于紧急医疗情况，建议立即就医。使用专业但通俗易懂的中文回答。"
)


# ─── Session CRUD ─────────────────────────────────────────────────────────────


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = Session(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        title=payload.title,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == current_user.id)
        .order_by(Session.updated_at.desc())
    )
    return [SessionResponse.model_validate(s) for s in result.scalars().all()]


@router.get("/sessions/{session_id}", response_model=SessionWithMessages)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_or_404(session_id, current_user.id, db)
    messages = await load_session_messages(session_id, db)
    resp = SessionWithMessages.model_validate(session)
    resp.messages = [MessageResponse.model_validate(m) for m in messages]
    return resp


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await _get_session_or_404(session_id, current_user.id, db)
    await redis_client.invalidate_session(session_id)
    await db.delete(session)
    await db.commit()


# ─── Streaming Chat ───────────────────────────────────────────────────────────


@router.get("/sessions/{session_id}/stream")
async def stream_chat(
    session_id: str,
    query: str,
    request: Request,
    top_k: int = 5,
    candidate_k: int = 20,
    use_rerank: bool = True,
    use_hybrid: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """SSE endpoint. The client passes the query as a query-param and listens for events."""
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=422, detail="查询不能为空")
    if len(query) > 2000:
        raise HTTPException(status_code=422, detail="查询过长（最大2000字符）")

    session = await _get_session_or_404(session_id, current_user.id, db)

    # Load history & possibly summarise
    messages = await load_session_messages(session_id, db)
    session_summary = await maybe_summarise(session, messages, db)
    chat_history = build_lc_messages(messages, session_summary, _SYSTEM_PROMPT_BASE)

    # Save user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=query.strip(),
    )
    db.add(user_msg)
    await db.commit()
    await redis_client.invalidate_session(session_id)  # invalidate so next load is fresh

    final_answer_parts: list[str] = []
    last_sources: list[Any] = []
    last_trace_steps: list[dict] = []

    async def event_generator() -> AsyncGenerator[str, None]:
        nonlocal last_sources
        try:
            async for event in stream_agent(
                query=query.strip(),
                chat_history=chat_history[1:],  # exclude system
                db=db,
                session_id=session_id,
                session_summary=session_summary,
                top_k=top_k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
                use_hybrid=use_hybrid,
            ):
                if await request.is_disconnected():
                    break

                ev_type = event.get("type", "")

                if ev_type == "token":
                    final_answer_parts.append(event.get("content", ""))
                elif ev_type == "sources":
                    last_sources = event.get("data", [])
                elif ev_type == "step":
                    last_trace_steps.append(
                        {"step": event.get("step"), "data": event.get("data", {})}
                    )

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("SSE stream error")
            err_event = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(err_event, ensure_ascii=False)}\n\n"
        finally:
            # Persist assistant message
            final_answer = "".join(final_answer_parts)
            if final_answer:
                metadata: dict[str, Any] = {
                    "sources": last_sources[:10],
                    "steps": last_trace_steps,
                }
                assistant_msg = Message(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=final_answer,
                    msg_metadata=metadata,
                )
                db.add(assistant_msg)
                # 从第一条用户消息更新会话标题
                if session.title == "新对话" and query:
                    session.title = query[:50]
                    db.add(session)
                await db.commit()
                await redis_client.invalidate_session(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def post_message(
    session_id: str,
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Non-streaming chat endpoint (returns full answer at once)."""
    session = await _get_session_or_404(session_id, current_user.id, db)

    messages = await load_session_messages(session_id, db)
    session_summary = await maybe_summarise(session, messages, db)
    chat_history = build_lc_messages(messages, session_summary, _SYSTEM_PROMPT_BASE)

    user_msg = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=payload.query,
    )
    db.add(user_msg)
    await db.commit()

    final_answer = ""
    all_sources: list = []
    steps: list = []

    async for event in stream_agent(
        query=payload.query,
        chat_history=chat_history[1:],
        db=db,
        session_id=session_id,
        session_summary=session_summary,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        use_rerank=payload.use_rerank,
        use_hybrid=payload.use_hybrid,
    ):
        if event.get("type") == "token":
            final_answer += event.get("content", "")
        elif event.get("type") == "sources":
            all_sources = event.get("data", [])
        elif event.get("type") == "step":
            steps.append({"step": event.get("step"), "data": event.get("data", {})})
        elif event.get("type") == "done":
            if not final_answer:
                final_answer = event.get("data", {}).get("final_answer", "")

    ass_msg = Message(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=final_answer,
        msg_metadata={"sources": all_sources[:10], "steps": steps},
    )
    db.add(ass_msg)
    if session.title == "新对话":
        session.title = payload.query[:50]
        db.add(session)
    await db.commit()
    await db.refresh(ass_msg)
    await redis_client.invalidate_session(session_id)

    return MessageResponse.model_validate(ass_msg)


# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _get_session_or_404(session_id: str, user_id: str, db: AsyncSession) -> Session:
    result = await db.execute(
        select(Session).where(Session.id == session_id, Session.user_id == user_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session
