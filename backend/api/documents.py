"""Document upload and management routes.

Upload flow
-----------
1. Validate file size / type.
2. Create a Document row with status=processing.
3. Extract text (PDF via pdfplumber, plain-text passthrough).
4. Build hierarchical chunks (L1/L2/L3).
5. Generate dense + sparse vectors for L3 chunks.
6. Insert L3 chunks into Milvus.
7. Persist L1/L2 parent chunks into PostgreSQL via DocStore.
8. (Re-)fit BM25 on all L3 texts from this document.
9. Mark document as ready.
"""

from __future__ import annotations

import io
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.dependencies import get_current_user
from ..config import get_settings
from ..database import AsyncSessionFactory, get_db
from ..models.document import Document, ParentChunk
from ..models.user import User
from ..rag.chunker import Chunk, build_hierarchical_chunks
from ..rag.doc_store import doc_store
from ..rag.embedder import bm25_manager, embed_dense
from ..rag.milvus_store import delete_by_doc_id, insert_chunks
from ..schemas.document import DocumentResponse

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/documents", tags=["documents"])

_ALLOWED_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/octet-stream",
}


def _extract_text(content: bytes, content_type: str) -> str:
    if "pdf" in content_type:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF 解析失败: {e}",
            )
    return content.decode("utf-8", errors="replace")


async def _process_document(doc_id: str, text: str, filename: str) -> None:
    """后台任务：分块 → 嵌入 → 存储 → BM25 训练。

    注意：后台任务必须自建 Session，不能复用请求的 Session（请求结束后 Session 已关闭）。
    """
    async with AsyncSessionFactory() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            return

        try:
            # ── 第一步：文本分块 ──────────────────────────────────────────────
            hierarchy = build_hierarchical_chunks(text, doc_id)
            l1_chunks: list[Chunk] = hierarchy["l1"]
            l2_chunks: list[Chunk] = hierarchy["l2"]
            l3_chunks: list[Chunk] = hierarchy["l3"]

            if not l3_chunks:
                doc.status = "error"
                doc.error_msg = "文档内容为空或无法切分"
                db.add(doc)
                await db.commit()
                return

            # ── 第二步：生成 L3 稠密向量 ──────────────────────────────────────
            l3_texts = [c.content for c in l3_chunks]
            BATCH = 10  # DashScope text-embedding-v3 单批最多 10 条
            dense_vecs: list[list[float]] = []
            for i in range(0, len(l3_texts), BATCH):
                batch_vecs = await embed_dense(l3_texts[i : i + BATCH])
                dense_vecs.extend(batch_vecs)

            # ── 第三步：BM25 训练与稀疏向量 ───────────────────────────────────
            await bm25_manager.fit_and_save(l3_texts)
            sparse_vecs = bm25_manager.encode_documents(l3_texts)

            # ── 第四步：写入 Milvus ───────────────────────────────────────────
            meta = {"filename": filename, "doc_id": doc_id}
            insert_chunks(
                chunk_ids=[c.chunk_id for c in l3_chunks],
                doc_id=doc_id,
                texts=l3_texts,
                dense_vectors=dense_vecs,
                sparse_vectors=sparse_vecs,
                parent_l2_ids=[c.parent_l2_id or "" for c in l3_chunks],
                parent_l1_ids=[c.parent_l1_id or "" for c in l3_chunks],
                chunk_indices=[c.chunk_index for c in l3_chunks],
                meta=meta,
            )

            # ── 第五步：写入 PostgreSQL 父块 ──────────────────────────────────
            for c in l1_chunks + l2_chunks:
                pc = ParentChunk(
                    id=c.chunk_id,
                    doc_id=c.doc_id,
                    level=c.level,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    total_children=c.total_children,
                    parent_id=c.parent_l1_id,
                    chunk_metadata=meta,
                )
                await doc_store.save_chunk(pc, db)

            doc.total_chunks = len(l3_chunks)
            doc.status = "ready"
            doc.error_msg = None
            db.add(doc)
            await db.commit()
            logger.info(
                "文档 %s 处理完成：%d 个 L3 分块，%d 个 L1，%d 个 L2",
                doc_id, len(l3_chunks), len(l1_chunks), len(l2_chunks),
            )

        except Exception as e:
            logger.exception("文档处理失败：%s", doc_id)
            try:
                doc.status = "error"
                doc.error_msg = str(e)[:500]
                db.add(doc)
                await db.commit()
            except Exception:
                pass


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content_type = file.content_type or "application/octet-stream"
    if content_type not in _ALLOWED_TYPES and not content_type.startswith("text/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的文件类型: {content_type}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大（最大 {settings.MAX_UPLOAD_SIZE_MB} MB）",
        )

    # Delete existing document with same filename for this user (dedup)
    existing = await db.execute(
        select(Document).where(
            Document.filename == file.filename,
            Document.user_id == current_user.id,
        )
    )
    old_doc = existing.scalar_one_or_none()
    if old_doc:
        try:
            delete_by_doc_id(old_doc.id)
        except Exception:
            pass
        old_chunk_ids = await doc_store.delete_by_doc(old_doc.id, db)
        from ..cache.redis_client import redis_client
        await redis_client.invalidate_doc_chunks(old_doc.id, old_chunk_ids)
        await db.delete(old_doc)
        await db.commit()

    doc_id = str(uuid.uuid4())
    text = _extract_text(content, content_type)

    doc = Document(
        id=doc_id,
        filename=file.filename or "unknown",
        file_size=len(content),
        content_type=content_type,
        user_id=current_user.id,
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(_process_document, doc_id, text, file.filename or "unknown")

    return DocumentResponse.model_validate(doc)


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    return [DocumentResponse.model_validate(d) for d in result.scalars().all()]


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentResponse.model_validate(doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    try:
        delete_by_doc_id(doc_id)
    except Exception:
        pass

    chunk_ids = await doc_store.delete_by_doc(doc_id, db)
    from ..cache.redis_client import redis_client
    await redis_client.invalidate_doc_chunks(doc_id, chunk_ids)

    await db.delete(doc)
    await db.commit()


# ─── 医学图片上传（多模态知识图谱）──────────────────────────────────────────

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.post("/images", status_code=status.HTTP_201_CREATED)
async def upload_medical_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    entity_name: str = "",
    entity_label: str = "Disease",
    current_user: User = Depends(get_current_user),
):
    """
    上传医学图片并关联到知识图谱实体。

    流程：CLIP 编码 → 创建 Image 节点 → DEPICTS 关系连接到实体。
    """
    content_type = file.content_type or ""
    if content_type not in _IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"不支持的图片类型: {content_type}，支持 JPEG/PNG/WebP/BMP",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大（最大 {settings.MAX_UPLOAD_SIZE_MB} MB）",
        )

    import os
    from pathlib import Path

    upload_dir = Path(settings.IMAGE_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    image_id = str(uuid.uuid4())
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    file_path = upload_dir / f"{image_id}{ext}"
    file_path.write_bytes(content)

    background_tasks.add_task(
        _process_image, image_id, str(file_path), entity_name, entity_label
    )

    return {
        "image_id": image_id,
        "file_path": str(file_path),
        "status": "processing",
        "entity_name": entity_name or None,
    }


async def _process_image(
    image_id: str,
    file_path: str,
    entity_name: str,
    entity_label: str,
) -> None:
    """后台任务：CLIP 编码 → 写入 Neo4j Image 节点 → 关联实体。"""
    try:
        from ..kg.image_encoder import encode_image
        from ..kg.graph_store import upsert_image_node, link_image_to_entity

        clip_vec = encode_image(file_path)

        await upsert_image_node(
            image_id=image_id,
            file_path=file_path,
            clip_embedding=clip_vec,
            metadata={"original_entity": entity_name},
        )

        if entity_name:
            await link_image_to_entity(
                image_id=image_id,
                entity_label=entity_label,
                entity_name=entity_name,
            )

        logger.info("Image %s processed and linked to %s:%s", image_id, entity_label, entity_name)
    except Exception:
        logger.exception("Image processing failed: %s", image_id)


# ─── 文档知识图谱抽取 ─────────────────────────────────────────────────────────


@router.post("/{doc_id}/extract-kg", status_code=status.HTTP_202_ACCEPTED)
async def extract_document_kg(
    doc_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """对已上传的文档执行知识图谱实体关系抽取。"""
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="文档尚未处理完成")

    background_tasks.add_task(_extract_kg_from_doc, doc_id)
    return {"doc_id": doc_id, "status": "kg_extracting"}


async def _extract_kg_from_doc(doc_id: str) -> None:
    """后台任务：从文档分块中抽取知识图谱三元组。"""
    try:
        from ..kg.entity_extractor import extract_and_persist
        from ..kg.graph_store import init_graph_schema

        await init_graph_schema()

        async with AsyncSessionFactory() as db:
            result = await db.execute(
                select(ParentChunk)
                .where(ParentChunk.doc_id == doc_id, ParentChunk.level == 1)
                .order_by(ParentChunk.chunk_index)
            )
            chunks = list(result.scalars().all())

            for chunk in chunks:
                if chunk.content and len(chunk.content.strip()) > 50:
                    await extract_and_persist(chunk.content, doc_id=doc_id)

        logger.info("KG extraction complete for doc %s (%d chunks)", doc_id, len(chunks))
    except Exception:
        logger.exception("KG extraction failed for doc %s", doc_id)
