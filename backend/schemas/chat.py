from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=5, le=50)
    use_rerank: bool = True
    use_hybrid: bool = True


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    msg_metadata: Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", max_length=256)


class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionWithMessages(SessionResponse):
    messages: list[MessageResponse] = []


class StreamEvent(BaseModel):
    """SSE event payload from the streaming chat endpoint."""

    type: str  # token | thinking | step | sources | tool_start | tool_end | error | done
    content: Optional[str] = None
    step: Optional[str] = None
    data: Optional[dict[str, Any]] = None
