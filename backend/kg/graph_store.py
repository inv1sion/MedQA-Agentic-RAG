"""
Neo4j 图数据库存储层。

管理医学知识图谱的实体与关系 CRUD，支持以下实体类型：
  Disease（疾病）、Symptom（症状）、Drug（药物）、
  BodyPart（身体部位）、Procedure（检查/治疗手段）、Image（医学图像）。

关系示例：
  (Disease)-[:HAS_SYMPTOM]->(Symptom)
  (Disease)-[:TREATED_BY]->(Drug)
  (Disease)-[:AFFECTS]->(BodyPart)
  (Disease)-[:DIAGNOSED_BY]->(Procedure)
  (Image)-[:DEPICTS]->(Disease)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── 单例驱动 ──────────────────────────────────────────────────────────────────
_driver: Optional[AsyncDriver] = None


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


# ── 初始化约束与索引 ──────────────────────────────────────────────────────────


async def init_graph_schema() -> None:
    """创建唯一性约束和全文索引（幂等操作）。"""
    driver = await get_driver()
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BodyPart) REQUIRE b.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Procedure) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE",
    ]
    async with driver.session() as session:
        for cypher in constraints:
            await session.run(cypher)
        # 全文索引用于模糊搜索
        await session.run(
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (n:Disease|Symptom|Drug|BodyPart|Procedure) ON EACH [n.name, n.aliases]"
        )
    logger.info("Neo4j graph schema initialized.")


# ── 实体写入 ──────────────────────────────────────────────────────────────────


async def upsert_entity(
    label: str,
    name: str,
    properties: Optional[dict[str, Any]] = None,
) -> None:
    """MERGE 一个实体节点（不存在则创建，存在则更新属性）。"""
    driver = await get_driver()
    props = properties or {}
    cypher = (
        f"MERGE (n:{label} {{name: $name}}) "
        f"SET n += $props"
    )
    async with driver.session() as session:
        await session.run(cypher, name=name, props=props)


async def upsert_image_node(
    image_id: str,
    file_path: str,
    clip_embedding: list[float],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """MERGE 一个 Image 节点，包含 CLIP 嵌入向量与文件路径。"""
    driver = await get_driver()
    props = {
        "file_path": file_path,
        "clip_embedding": clip_embedding,
        **(metadata or {}),
    }
    cypher = (
        "MERGE (i:Image {image_id: $image_id}) "
        "SET i += $props"
    )
    async with driver.session() as session:
        await session.run(cypher, image_id=image_id, props=props)


# ── 关系写入 ──────────────────────────────────────────────────────────────────


async def upsert_relation(
    from_label: str,
    from_name: str,
    rel_type: str,
    to_label: str,
    to_name: str,
    properties: Optional[dict[str, Any]] = None,
) -> None:
    """MERGE 两个实体之间的关系。"""
    driver = await get_driver()
    props = properties or {}
    cypher = (
        f"MERGE (a:{from_label} {{name: $from_name}}) "
        f"MERGE (b:{to_label} {{name: $to_name}}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        f"SET r += $props"
    )
    async with driver.session() as session:
        await session.run(cypher, from_name=from_name, to_name=to_name, props=props)


async def link_image_to_entity(
    image_id: str,
    entity_label: str,
    entity_name: str,
    rel_type: str = "DEPICTS",
) -> None:
    """将 Image 节点关联到实体节点。"""
    driver = await get_driver()
    cypher = (
        "MERGE (i:Image {image_id: $image_id}) "
        f"MERGE (e:{entity_label} {{name: $entity_name}}) "
        f"MERGE (i)-[r:{rel_type}]->(e)"
    )
    async with driver.session() as session:
        await session.run(cypher, image_id=image_id, entity_name=entity_name)


# ── 查询 ──────────────────────────────────────────────────────────────────────


async def search_entity(query: str, limit: int = 10) -> list[dict]:
    """全文搜索实体节点。"""
    driver = await get_driver()
    cypher = (
        "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) "
        "YIELD node, score "
        "RETURN labels(node) AS labels, node.name AS name, score "
        "ORDER BY score DESC LIMIT $limit"
    )
    async with driver.session() as session:
        result = await session.run(cypher, query=query, limit=limit)
        return [dict(record) async for record in result]


async def get_entity_relations(
    name: str,
    max_depth: int = 2,
    limit: int = 50,
) -> list[dict]:
    """获取实体的关系子图（BFS 拓展到 max_depth 跳）。"""
    driver = await get_driver()
    cypher = (
        "MATCH (n {name: $name})-[r*1..$max_depth]-(m) "
        "WITH n, r, m LIMIT $limit "
        "UNWIND r AS rel "
        "RETURN DISTINCT "
        "  startNode(rel).name AS from_name, labels(startNode(rel)) AS from_labels, "
        "  type(rel) AS rel_type, "
        "  endNode(rel).name AS to_name, labels(endNode(rel)) AS to_labels"
    )
    async with driver.session() as session:
        result = await session.run(
            cypher, name=name, max_depth=max_depth, limit=limit
        )
        return [dict(record) async for record in result]


async def get_images_for_entity(entity_name: str, limit: int = 5) -> list[dict]:
    """获取与某个实体关联的所有图片节点。"""
    driver = await get_driver()
    cypher = (
        "MATCH (i:Image)-[:DEPICTS]->(e {name: $name}) "
        "RETURN i.image_id AS image_id, i.file_path AS file_path, "
        "       i.clip_embedding AS clip_embedding "
        "LIMIT $limit"
    )
    async with driver.session() as session:
        result = await session.run(cypher, name=entity_name, limit=limit)
        return [dict(record) async for record in result]
