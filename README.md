# MEDQA — 中文医疗问答 Agentic RAG 系统

基于 LangChain Agent + Milvus 混合检索 + FastAPI 的中文医疗知识库问答平台。

---

## 目录

1. [系统架构](#系统架构)
2. [环境要求](#环境要求)
3. [快速启动](#快速启动)
4. [环境变量说明](#环境变量说明)
5. [API 接口一览](#api-接口一览)
6. [前端使用说明](#前端使用说明)
7. [文档处理链路](#文档处理链路)
8. [三级分块与 Auto-merging](#三级分块与-auto-merging)
9. [混合检索与重排序](#混合检索与重排序)
10. [会话记忆机制](#会话记忆机制)
11. [RAG 评估体系](#rag-评估体系)
12. [常见问题](#常见问题)

---

## 系统架构

```
前端 (Vue 3 CDN 单页)
    │  SSE 流式输出 / AbortController
    ▼
FastAPI 后端
    ├── JWT 鉴权 + RBAC (admin/user)
    ├── LangChain Agent (工具调用)
    │       ├── medqa_rag_search  ← 核心知识库检索工具
    │       └── hospital_query   ← 医院信息查询示例工具
    ├── RAG 流水线
    │       ├── 查询重写 (Step-Back / HyDE)
    │       ├── Milvus 混合检索 (Dense + BM25 / RRF)
    │       ├── Auto-merging (L3→L2→L1)
    │       ├── 相关性评分门控
    │       └── Qwen3 Rerank 精排
    └── asyncio.Queue 实时步骤推送
         (Searching → Grading → Rewriting)

存储层
    ├── PostgreSQL  — 用户、文档元数据、父块、会话、消息
    ├── Milvus      — L3 叶子块稠密/稀疏向量
    └── Redis       — 热点会话缓存、父文档缓存
```

---

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | 3.10 ~ 3.12 |
| Docker & Docker Compose | 20.10+ |
| WSL2 (Windows 用户) | Ubuntu 20.04+ |
| 阿里云 DashScope API Key | 用于 LLM / Embedding / Rerank |

---

## 快速启动

### 第一步：启动基础服务

```bash
cd /home/zhuanz1/projects/MEDQA
docker compose up -d
```

等待所有容器健康（约 30 秒）：

```bash
docker compose ps
# postgres、redis、milvus 均显示 healthy 即可
```

### 第二步：配置环境变量

```bash
cd backend
cp .env.example .env   # 若无此文件则手动创建，见下方"环境变量说明"
```

编辑 `.env`，至少填写：

```env
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx
RERANK_API_KEY=sk-xxxxxxxxxxxxxxxx
JWT_SECRET=你自己的随机密钥字符串
```

### 第三步：安装 Python 依赖

```bash
# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate      # Linux/WSL
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 第四步：启动 FastAPI 后端

```bash
cd /home/zhuanz1/projects/MEDQA/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后控制台显示：

```
数据库表创建完成
Milvus 集合初始化完成
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 第五步：访问前端

直接用浏览器打开：

```
http://localhost:8000
```

或直接打开文件：

```
frontend/index.html
```

> **注意**：直接打开文件时，前端 API 请求默认指向 `http://localhost:8000`，需确保后端正在运行。

---

## 环境变量说明

在 `backend/.env` 中配置：

```env
# ── 大语言模型（阿里云 DashScope / 兼容 OpenAI 格式）────────────────────────
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL=qwen-plus           # 可换 qwen-max / qwen-turbo 等
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096

# ── 嵌入向量模型 ─────────────────────────────────────────────────────────────
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIM=1024

# ── 重排序模型（Qwen3 Rerank / GTE-Rerank）──────────────────────────────────
RERANK_API_KEY=sk-xxxxxxxxxxxxxxxx
RERANK_MODEL=gte-rerank
RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_TOP_N=5
RERANK_CANDIDATE_K=20

# ── Milvus ───────────────────────────────────────────────────────────────────
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=                  # 本地部署留空即可
MILVUS_COLLECTION=medqa_chunks

# ── PostgreSQL ───────────────────────────────────────────────────────────────
POSTGRES_URL=postgresql+asyncpg://medqa:medqa123@localhost:5432/medqa

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
REDIS_SESSION_TTL=3600         # 会话缓存过期时间（秒）
REDIS_DOC_TTL=7200             # 父文档缓存过期时间（秒）

# ── JWT ──────────────────────────────────────────────────────────────────────
JWT_SECRET=修改为随机长字符串   # 生产环境必须修改！
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# ── 分块配置（与附图一致）────────────────────────────────────────────────────
L1_CHUNK_SIZE=1200
L1_CHUNK_OVERLAP=240
L2_CHUNK_SIZE=600
L2_CHUNK_OVERLAP=120
L3_CHUNK_SIZE=300
L3_CHUNK_OVERLAP=60

# ── 检索配置 ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K=5              # 最终返回 top-k
RETRIEVAL_CANDIDATE_K=20       # 召回候选数量
RELEVANCE_SCORE_THRESHOLD=0.5  # 相关性评分门控阈值
MERGE_THRESHOLD=0.5            # Auto-merging 合并触发比例

# ── 会话摘要 ─────────────────────────────────────────────────────────────────
SUMMARY_THRESHOLD=20           # 超过 20 条消息触发摘要
SUMMARY_WINDOW=10              # 每次摘要最近 10 条

# ── 应用全局 ─────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
MAX_UPLOAD_SIZE_MB=50
```

---

## API 接口一览

后端启动后可访问交互式文档：`http://localhost:8000/docs`

### 认证

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/api/auth/register` | 用户注册 | 公开 |
| POST | `/api/auth/login` | 登录，返回 JWT token | 公开 |
| GET  | `/api/auth/me` | 获取当前用户信息 | 登录用户 |

注册请求体示例：
```json
{
  "username": "zhangsan",
  "password": "your_password",
  "email": "zhangsan@example.com"
}
```

登录返回：
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

### 文档管理

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/api/documents/upload` | 上传文档（PDF/DOCX/TXT） | 登录用户 |
| GET  | `/api/documents/` | 获取文档列表 | 登录用户 |
| DELETE | `/api/documents/{id}` | 删除文档及其所有向量 | Admin |

上传文档（multipart/form-data）：
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@医疗指南.pdf"
```

### 会话与聊天

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/api/chat/sessions` | 创建新会话 | 登录用户 |
| GET  | `/api/chat/sessions` | 获取会话列表 | 登录用户 |
| GET  | `/api/chat/sessions/{id}/messages` | 获取会话历史消息 | 登录用户 |
| GET  | `/api/chat/stream` | SSE 流式问答 | 登录用户 |
| DELETE | `/api/chat/sessions/{id}` | 删除会话 | 登录用户 |

SSE 流式问答（GET + QueryString）：
```
GET /api/chat/stream?session_id=<uuid>&question=糖尿病的诊断标准是什么
Authorization: Bearer <token>
```

SSE 事件格式：
```
data: {"type": "step", "content": "🔍 正在检索知识库..."}
data: {"type": "token", "content": "根据"}
data: {"type": "token", "content": "《中国2型糖尿病防治指南》"}
data: {"type": "sources", "content": [...]}
data: {"type": "done"}
```

### 管理员接口

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| GET  | `/api/admin/users` | 获取所有用户列表 | Admin 专属 |
| PATCH | `/api/admin/users/{id}/role` | 修改用户角色 | Admin 专属 |

### RAG 评估

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/api/eval/run` | 运行评估集，返回 P@k/R@k/MRR/NDCG | Admin |
| GET  | `/api/eval/rerank-strategy` | Rerank 策略对比评估 | Admin |

---

## 前端使用说明

### 注册/登录
1. 打开页面，点击右上角「注册」填写用户名与密码
2. 登录后 JWT token 自动保存在 `localStorage`，页面刷新保持登录态

### 上传医疗文档
1. 点击左侧「文档管理」→「上传文档」
2. 支持 PDF、DOCX、TXT，单文件上限 50 MB（可在 `.env` 调整）
3. 重复上传同名文档会自动清理旧的向量数据再重新入库

### 开始问答
1. 点击「新建会话」
2. 在输入框输入中文医疗问题，回车或点击发送
3. 页面实时展示：
   - **思考步骤**（🔍 检索中 → ✅ 评分 → 🔄 重写）
   - **逐 token 打字机输出**
   - **引用来源**（可展开查看 rerank_score、文档名、分块内容）
4. 点击「停止」可随时中断生成

### RAG 过程面板
每次回答结束后点击「查看检索详情」可展开：
- 检索到的分块列表及 rerank_score
- 是否触发了查询重写（Step-Back / HyDE）
- Auto-merging 合并情况（L3→L2→L1）

---

## 文档处理链路

```
用户上传文件
    │
    ▼
文本提取（PDF/DOCX/TXT → 纯文本）
    │
    ▼
三级滑动窗口分块
    ├── L1：1200 字符，重叠 240（根级，粗粒度）
    ├── L2：600 字符，重叠 120（中粒度）
    └── L3：300 字符，重叠 60（叶子级，细粒度）
    │
    ▼
稠密向量生成（text-embedding-v3，1024 维）
    +
BM25 稀疏向量生成（基于语料训练的 jieba 分词）
    │
    ▼
写入存储
    ├── L3 块 → Milvus（稠密 + 稀疏向量）
    └── L1/L2 块 → PostgreSQL DocStore（父块原文）
    │
    ▼
Redis 缓存父文档 ID 映射（TTL 7200s）
```

---

## 三级分块与 Auto-merging

检索时优先在 L3 精细粒度命中，若多个 L3 块归属同一父块且覆盖率超过阈值（默认 50%），自动合并上升到父级：

```
检索召回 L3 候选
    │
    ├── 同一 L2 父块的 L3 子块数 / 该 L2 总子块数 ≥ MERGE_THRESHOLD
    │       → 替换为 L2 父块内容
    │
    └── 同一 L1 父块的 L2 子块数 / 该 L1 总子块数 ≥ MERGE_THRESHOLD
            → 替换为 L1 父块内容（最大上下文）
```

`MERGE_THRESHOLD` 可在 `.env` 中调整（默认 `0.5`）。

---

## 混合检索与重排序

```
用户查询
    │
    ├── [可选] 查询重写
    │       ├── Step-Back：抽象成更通用问题
    │       └── HyDE：生成假设答案再检索
    │
    ▼
Milvus Hybrid Search
    ├── 稠密检索（ANN，text-embedding-v3）
    └── BM25 稀疏检索（关键词匹配）
        └── RRF 融合排序（Reciprocal Rank Fusion）
    │
    ▼  失败时自动降级为纯稠密检索
    │
    ▼
Auto-merging（L3→L2→L1）
    │
    ▼
相关性评分门控（LLM 结构化输出，score < threshold 触发重写）
    │
    ▼
Qwen3 Rerank 精排（GTE-Rerank API，返回 rerank_score）
    │
    ▼
Top-K 结果传入 Agent 生成回答
```

**双向降级策略**：
- BM25 稀疏向量生成失败 → 只用稠密向量
- Milvus Hybrid Search 失败 → 降级为纯稠密 ANN 检索
- Rerank API 超时/失败 → 直接使用 RRF 排序结果

---

## 会话记忆机制

- 每条用户消息和 AI 回答都持久化到 PostgreSQL
- Redis 缓存最近会话的消息列表（TTL 1 小时）
- 当会话消息总数超过 `SUMMARY_THRESHOLD`（默认 20 条）时，自动对最早的 `SUMMARY_WINDOW` 条消息做 LLM 摘要，注入系统提示，防止 token 膨胀

---

## RAG 评估体系

在 `http://localhost:8000/docs` 中调用 `/api/eval/run`，传入评估集：

```json
{
  "questions": ["糖尿病的诊断标准？", "高血压如何分级？"],
  "ground_truth_ids": [["chunk_001", "chunk_003"], ["chunk_010"]],
  "top_k": 5,
  "candidate_k": 20
}
```

返回指标：

```json
{
  "precision_at_k": 0.82,
  "recall_at_k": 0.76,
  "mrr": 0.89,
  "ndcg": 0.84,
  "rerank_strategy": {
    "top_k": 5,
    "candidate_k": 20,
    "recall_candidate_ratio": 4.0
  }
}
```

---

## 常见问题

**Q：启动时报 `cannot connect to PostgreSQL`？**
> 确认 Docker 容器已健康：`docker compose ps`，PostgreSQL 端口 5432 已映射。

**Q：Milvus 集合创建失败？**
> Milvus 冷启动较慢，等待约 30 秒后重启后端：`uvicorn main:app ...`

**Q：上传 PDF 后中文乱码？**
> PDF 需包含可提取文本层。扫描版 PDF 需先用 OCR 工具（如 PaddleOCR）预处理。

**Q：BM25 模型文件 `bm25_model.pkl` 在哪里？**
> 首次上传文档后自动生成于 `backend/bm25_model.pkl`，无需手动创建。

**Q：如何创建第一个 Admin 账户？**
> 注册后在 PostgreSQL 直接更新角色：
> ```sql
> UPDATE users SET role='admin' WHERE username='your_username';
> ```

**Q：前端 SSE 连接被 Nginx 断开？**
> 在 Nginx 配置中添加：
> ```nginx
> proxy_read_timeout 300s;
> proxy_buffering off;
> X-Accel-Buffering: no;
> ```
