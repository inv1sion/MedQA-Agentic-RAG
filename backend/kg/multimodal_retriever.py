"""
多模态知识图谱检索器。

支持三种检索模式：
  1. 文本 → 图谱实体 + 关系推理（text2kg）
  2. 文本 → 匹配图片（text2image，CLIP 跨模态）
  3. 图片 → 匹配疾病实体（image2entity，CLIP 跨模态 + 图谱关系拓展）

检索结果可与现有 RAG 文本检索结果合并，为 Agent 提供更丰富的上下文。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from . import graph_store
from .image_encoder import compute_similarity, encode_image, encode_text, rank_by_similarity

logger = logging.getLogger(__name__)


async def text_to_kg(
    query: str,
    max_depth: int = 2,
    limit: int = 30,
) -> dict:
    """
    文本查询 → 图谱实体检索 + 关系子图拓展。

    流程：
      1. 全文搜索命中实体
      2. 对每个命中实体做 BFS 拓展获取关系子图
    """
    # 第一步：全文检索命中实体
    entities = await graph_store.search_entity(query, limit=10)

    if not entities:
        return {"entities": [], "relations": [], "images": []}

    # 第二步：对 top 实体拓展关系子图
    all_relations = []
    all_images = []
    seen_rels = set()

    for entity in entities[:3]:  # 拓展 top-3 实体
        name = entity["name"]

        # 关系子图
        rels = await graph_store.get_entity_relations(name, max_depth=max_depth, limit=limit)
        for r in rels:
            key = (r["from_name"], r["rel_type"], r["to_name"])
            if key not in seen_rels:
                seen_rels.add(key)
                all_relations.append(r)

        # 关联图片
        images = await graph_store.get_images_for_entity(name, limit=3)
        all_images.extend(images)

    return {
        "entities": entities,
        "relations": all_relations,
        "images": all_images,
    }


async def text_to_images(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    文本查询 → CLIP 跨模态匹配图片。

    流程：
      1. 文本编码为 CLIP 向量
      2. 全文搜索图谱实体
      3. 收集关联图片节点
      4. 按 CLIP 相似度排序返回
    """
    # 文本 CLIP 编码
    text_vec = encode_text(query)

    # 从图谱获取候选图片
    entities = await graph_store.search_entity(query, limit=5)
    candidate_images = []
    seen_ids = set()

    for entity in entities:
        images = await graph_store.get_images_for_entity(entity["name"], limit=10)
        for img in images:
            if img["image_id"] not in seen_ids:
                seen_ids.add(img["image_id"])
                candidate_images.append(img)

    if not candidate_images:
        return []

    # CLIP 相似度排序
    return rank_by_similarity(text_vec, candidate_images, top_k=top_k)


async def image_to_entities(
    image_input: Union[str, Path],
    top_k: int = 5,
    relation_depth: int = 1,
) -> dict:
    """
    图片 → CLIP 编码 → 匹配图谱中最相似的 Image 节点 → 拓展关联疾病实体。

    流程：
      1. 输入图片编码为 CLIP 向量
      2. 从 Neo4j 获取所有 Image 节点的嵌入
      3. 按 CLIP 相似度排序
      4. 对匹配到的 Image 节点沿 DEPICTS 关系找到实体
      5. 拓展实体的关系子图
    """
    query_vec = encode_image(image_input)

    # 获取图谱中全部 Image 节点（生产环境应建向量索引）
    driver = await graph_store.get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (i:Image) "
            "WHERE i.clip_embedding IS NOT NULL "
            "RETURN i.image_id AS image_id, i.file_path AS file_path, "
            "       i.clip_embedding AS clip_embedding "
            "LIMIT 500"
        )
        all_images = [dict(record) async for record in result]

    if not all_images:
        return {"matched_images": [], "entities": [], "relations": []}

    # CLIP 排序
    matched = rank_by_similarity(query_vec, all_images, top_k=top_k)

    # 沿 DEPICTS 关系拓展到实体
    linked_entities = []
    all_relations = []
    seen_entities = set()

    for img in matched:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (i:Image {image_id: $iid})-[:DEPICTS]->(e) "
                "RETURN labels(e) AS labels, e.name AS name",
                iid=img["image_id"],
            )
            async for record in result:
                name = record["name"]
                if name not in seen_entities:
                    seen_entities.add(name)
                    linked_entities.append(dict(record))

                    # 拓展关系
                    rels = await graph_store.get_entity_relations(
                        name, max_depth=relation_depth, limit=20
                    )
                    all_relations.extend(rels)

    return {
        "matched_images": matched,
        "entities": linked_entities,
        "relations": all_relations,
    }


def format_kg_context(kg_result: dict) -> str:
    """将图谱检索结果格式化为 LLM 可读的上下文字符串。"""
    parts = []

    # 实体
    entities = kg_result.get("entities", [])
    if entities:
        entity_strs = []
        for e in entities[:10]:
            labels = e.get("labels", [])
            label_str = "/".join(labels) if isinstance(labels, list) else str(labels)
            entity_strs.append(f"  - [{label_str}] {e.get('name', '')}")
        parts.append("【知识图谱 · 相关实体】\n" + "\n".join(entity_strs))

    # 关系
    relations = kg_result.get("relations", [])
    if relations:
        rel_strs = []
        for r in relations[:15]:
            rel_strs.append(
                f"  - {r.get('from_name', '')} --[{r.get('rel_type', '')}]--> {r.get('to_name', '')}"
            )
        parts.append("【知识图谱 · 关系】\n" + "\n".join(rel_strs))

    # 图片
    images = kg_result.get("matched_images", kg_result.get("images", []))
    if images:
        img_strs = []
        for img in images[:5]:
            sim = img.get("similarity", "")
            sim_str = f" (相似度: {sim:.3f})" if isinstance(sim, float) else ""
            img_strs.append(f"  - {img.get('file_path', img.get('image_id', ''))}{sim_str}")
        parts.append("【知识图谱 · 关联图片】\n" + "\n".join(img_strs))

    if not parts:
        return "知识图谱中未找到相关内容。"

    return "\n\n".join(parts)
