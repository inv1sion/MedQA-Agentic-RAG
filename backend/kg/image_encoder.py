"""
Chinese-CLIP 跨模态编码器。

使用 OFA-Sys/chinese-clip-vit-base-patch16 将图片和文本
映射到同一 512 维向量空间，支持：
  - 图片 → 向量（用于图片入图谱 + 图片检索）
  - 文本 → 向量（用于文本查图片）
  - 相似度计算（用于跨模态匹配）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── 懒加载模型单例 ────────────────────────────────────────────────────────────
_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return

    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    model_name = settings.CLIP_MODEL_NAME
    device = settings.CLIP_DEVICE

    logger.info("Loading Chinese-CLIP model: %s on %s", model_name, device)
    _processor = ChineseCLIPProcessor.from_pretrained(model_name)
    _model = ChineseCLIPModel.from_pretrained(model_name).to(device).eval()
    logger.info("Chinese-CLIP model loaded.")


def _get_device() -> str:
    return settings.CLIP_DEVICE


# ── 编码接口 ──────────────────────────────────────────────────────────────────


def encode_image(image_input: Union[str, Path, "PIL.Image.Image"]) -> list[float]:
    """
    将图片编码为 CLIP 向量。

    参数：
        image_input: 文件路径字符串/Path 对象，或 PIL.Image 对象
    返回：
        归一化后的向量 (list[float])
    """
    import torch
    from PIL import Image

    _load_model()

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    inputs = _processor(images=image, return_tensors="pt").to(_get_device())
    with torch.no_grad():
        features = _model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().tolist()


def encode_text(text: str) -> list[float]:
    """
    将文本编码为 CLIP 向量。

    参数：
        text: 中文文本
    返回：
        归一化后的向量 (list[float])
    """
    import torch

    _load_model()

    inputs = _processor(text=[text], padding=True, return_tensors="pt").to(_get_device())
    with torch.no_grad():
        features = _model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().tolist()


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个归一化向量的余弦相似度。"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b))


def rank_by_similarity(
    query_vec: list[float],
    candidates: list[dict],
    embedding_key: str = "clip_embedding",
    top_k: int = 5,
) -> list[dict]:
    """
    根据 CLIP 向量相似度对候选列表排序。

    参数：
        query_vec: 查询向量
        candidates: 候选字典列表，每个需包含 embedding_key 字段
        embedding_key: 嵌入向量的字段名
        top_k: 返回前 k 个
    返回：
        按相似度降序排列的候选列表（附加 similarity 字段）
    """
    scored = []
    for c in candidates:
        emb = c.get(embedding_key)
        if emb is None:
            continue
        sim = compute_similarity(query_vec, emb)
        scored.append({**c, "similarity": sim})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]
