---
name: medqa-dev
description: "MEDQA 中文医疗问答系统开发专家。Use when: 开发、调试、扩展 MEDQA 项目的 RAG 检索流水线、LangChain Agent 工具、评估框架、FastAPI 接口或 Milvus/PostgreSQL/Redis 基础设施。触发词：medqa / rag / rerank / milvus / agent / evaluator / 医疗问答 / 检索 / 重排 / 评估指标。"
argument-hint: "描述你想修改的模块，例如：'优化 reranker' / '新增工具' / '调试评估'"
---

# MEDQA 开发 Skill

## 项目概览

| 维度 | 详情 |
|------|------|
| **业务** | 中文医疗 RAG 问答系统，支持文档上传、智能检索、流式对话 |
| **后端** | FastAPI + LangGraph ReAct Agent + Milvus + PostgreSQL + Redis |
| **LLM** | 阿里云 DashScope `qwen-plus`（可通过 `.env` 替换） |
| **嵌入** | `text-embedding-v3`（1024 维，DashScope） |
| **重排** | `gte-rerank`（DashScope Rerank API） |
| **部署** | `docker-compose.yml` 管理 Milvus / PostgreSQL / Redis |

---

## 目录结构与职责

```
backend/
├── config.py            # Settings（pydantic-settings，读取 .env）
├── database.py          # AsyncSessionFactory, init_db()
├── main.py              # FastAPI 应用入口
├── agent/
│   ├── agent.py         # LangGraph ReAct Agent + SSE 流式推送
│   ├── tools.py         # medqa_rag_search / hospital_query 工具
│   └── memory.py        # 对话历史摘要
├── api/
│   ├── chat.py          # /chat SSE 接口
│   ├── documents.py     # 文档上传/管理
│   ├── auth.py          # 注册/登录
│   └── admin.py         # 管理员接口
├── rag/
│   ├── retriever.py     # 混合检索（dense + BM25 → RRF），降级为纯稠密
│   ├── reranker.py      # Qwen3 Rerank 精排
│   ├── grader.py        # LLM 相关性评分 + 重写判断
│   ├── query_rewriter.py# 查询重写
│   ├── auto_merger.py   # 相邻分块自动合并
│   ├── chunker.py       # L3 分块策略
│   ├── embedder.py      # 稠密嵌入 + BM25 管理器
│   ├── milvus_store.py  # Milvus CRUD + hybrid_search / dense_search
│   └── doc_store.py     # PostgreSQL 文档元数据
├── eval/
│   └── evaluator.py     # 评估框架：P@k / R@k / MRR / NDCG@k
└── auth/ cache/ models/ schemas/   # 认证、Redis、ORM 模型、Pydantic Schema
run_eval.py              # 评估一键运行入口
```

---

## 核心配置（`backend/config.py`）

所有配置通过 `.env` 文件注入，关键变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `qwen-plus` | 对话大模型 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `RERANK_TOP_N` | `5` | 重排后返回数 |
| `RERANK_CANDIDATE_K` | `20` | 重排前候选集大小 |
| `MILVUS_COLLECTION` | `medqa_chunks` | Milvus 集合名 |
| `POSTGRES_URL` | `postgresql+asyncpg://...` | 异步 PG 连接串 |

---

## RAG 流水线（`backend/rag/`）

```
query
  │
  ├─► retriever.retrieve(query, candidate_k=20, use_hybrid=True)
  │     └─ hybrid_search (dense + BM25 sparse → RRF) → 降级 dense_search
  │
  ├─► auto_merger.auto_merge(chunks)          # 合并相邻分块
  │
  ├─► reranker.rerank(query, chunks, top_n)   # Qwen3 Rerank 精排
  │
  ├─► grader.grade_documents(query, chunks)   # LLM 相关性打分
  │
  └─► query_rewriter.rewrite_query(query)     # 低质量时重写
```

**关键参数**：
- `candidate_k=20`：召回阶段候选数（固定）
- `top_k=5`：最终返回数（`--k` 参数控制，评估时默认 5）

---

## Agent 工具（`backend/agent/tools.py`）

| 工具名 | 描述 | 关键参数 |
|--------|------|----------|
| `medqa_rag_search` | 完整 RAG 流水线检索 | `query, top_k=5, use_rerank=True, use_hybrid=True` |
| `hospital_query` | 医院/科室查询（可扩展） | `query` |

**新增工具步骤**：
1. 在 `tools.py` 中定义 `Input(BaseModel)` + `_make_xxx_tool(db, queue)` 函数
2. 在 `build_tools(db, queue)` 列表中注册
3. Agent 自动发现，无需修改 `agent.py`

---

## 评估框架（`run_eval.py` + `backend/eval/evaluator.py`）

**运行方式**：
```bash
# 在项目根目录
python run_eval.py --n 50 --k 5 --out eval_results.json
```

**4 种对比策略**：

| ID | 策略 | `use_hybrid` | `use_rerank` |
|----|------|:---:|:---:|
| A  | 混合 + Rerank（默认） | ✓ | ✓ |
| B  | 混合，无 Rerank | ✓ | ✗ |
| C  | 纯稠密 + Rerank | ✗ | ✓ |
| D  | 纯稠密，无 Rerank | ✗ | ✗ |

**评估指标**：`Precision@k`、`Recall@k`、`MRR`、`NDCG@k`

**修改指标**：在 `evaluator.py` 的 `_compute_metrics()` 函数中添加新公式。

---

## 常见开发任务

### 新增 RAG 组件
1. 在 `backend/rag/` 创建新模块
2. 在 `tools.py` 的 `_make_rag_tool` 函数中插入调用步骤
3. 用 `_push_step(queue, "新步骤名", {...})` 推送前端可见的进度事件

### 修改检索策略
- 候选集大小：修改 `config.py` 中 `RERANK_CANDIDATE_K`
- 向量维度变化：同步修改 `EMBEDDING_DIM` 并重建 Milvus 集合

### 启动基础设施
```bash
docker compose up -d          # 启动 Milvus / PostgreSQL / Redis
uvicorn backend.main:app --reload  # 启动 FastAPI 开发服务器
```

### 调试评估（不启动全服务）
```bash
python run_eval.py --n 10 --k 3   # 快速小规模测试
```

---

## 技术约束

- **异步优先**：所有 DB / Milvus / HTTP 操作使用 `async/await`
- **工具线程安全**：LangGraph 工具在线程池运行，用 `_push_step()` 回传事件到 asyncio 事件循环
- **BM25 状态**：`bm25_manager.is_fitted` 为 False 时自动降级为纯稠密检索
- **API Key**：通过 `.env` 注入，不得硬编码
