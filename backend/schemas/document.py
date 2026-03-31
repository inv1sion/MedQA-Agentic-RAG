from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    content_type: str
    user_id: str
    status: str
    total_chunks: int
    error_msg: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkInfo(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    level: int
    rerank_score: Optional[float] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    parent_l2_id: Optional[str] = None
    parent_l1_id: Optional[str] = None
    merged_from: Optional[list[str]] = None
