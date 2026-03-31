#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量将后端文件中残留的英文注释替换为中文。"""

import re
from pathlib import Path

BASE = Path("/home/zhuanz1/projects/MEDQA/backend")

# ─── tools.py ────────────────────────────────────────────────────────────────
def fix_tools():
    p = BASE / "agent/tools.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ("# ── Step 2: Auto-merge", "# ── 第二步：自动合并"),
        ("# ── Step 3: Rerank",     "# ── 第三步：Rerank 重排"),
        ("# ── Step 4: Grade",      "# ── 第四步：相关性评分"),
        ("# ── Step 5: Rewrite if needed", "# ── 第五步：必要时重写查询"),
        ("# ── Step 6: Build sources", "# ── 第六步：构建来源列表"),
        ("# over-fetch for rerank", "# 过采样备重排"),
        ("# ─── Hospital Query Tool", "# ─── 医院查询工具"),
        ("\"\"\"Factory that closes over db/queue/doc_ids.\"\"\"",
         "\"\"\"db/queue/doc_ids 闭包工厂函数。\"\"\""),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [tools.py] 替换: {old[:40]!r}")
        else:
            print(f"  [tools.py] 未找到: {old[:40]!r}")
    p.write_text(text, encoding="utf-8")

# ─── memory.py ───────────────────────────────────────────────────────────────
def fix_memory():
    p = BASE / "agent/memory.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""Session memory management with automatic summarisation.\n\nFlow:\n  1. Load messages for a session (Redis → PostgreSQL).\n  2. When message count > SUMMARY_THRESHOLD, summarise the oldest SUMMARY_WINDOW\n     messages into a single assistant summary message and persist it.\n  3. Inject the summary as a system prompt prefix so the agent has long-range context\n     without inflating the token budget.\n"""',
         '"""\n会话记忆管理模块，支持自动摘要压缩。\n\n流程：\n  1. 从 Redis 或 PostgreSQL 加载会话消息。\n  2. 当消息数 > SUMMARY_THRESHOLD 时，对最旧的 SUMMARY_WINDOW 条消息\n     生成单条摘要并持久化。\n  3. 将摘要注入系统提示前缀，在不膨胀 token 预算的前提下维持长程上下文。\n"""'),
        ('"\"\"\"Call LLM to compress a list of messages into a short summary.\"\"\"',
         '"\"\"\"调用 LLM 将一批消息压缩为简短摘要。\"\"\"'),
        ('"""Call LLM to compress a list of messages into a short summary."""',
         '"""调用 LLM 将一批消息压缩为简短摘要。"""'),
        ('"""Load messages from Redis cache or PostgreSQL, ordered by creation time."""',
         '"""按创建时间顺序，从 Redis 缓存或 PostgreSQL 加载消息。"""'),
        ('# Try Redis first',   '# 优先查 Redis 缓存'),
        ('# Reconstruct lightweight objects for the in-memory path',
         '# 重建轻量对象供内存路径使用'),
        ('# Warm-up cache', '# 预热缓存'),
        ('# Summarise earliest window', '# 对最旧的一批消息做摘要'),
        ('# Merge with existing summary', '# 与已有摘要合并'),
        ('# Take the last N user/assistant messages', '# 取最近 N 条用户/助手消息'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [memory.py] 替换: {old[:50]!r}")
        else:
            print(f"  [memory.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── agent.py ────────────────────────────────────────────────────────────────
def fix_agent():
    p = BASE / "agent/agent.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""LangChain OpenAI-tools agent for MedQA.\n\nStreaming architecture\n----------------------\nThe agent is invoked via `agent_executor.astream(stream_mode="messages")`.\nEach yielded chunk is either a AIMessageChunk (token) or a ToolMessage/AIMessage\nwith tool calls. We intercept both to push SSE events.\n\nReal-time tool-step visibility\n------------------------------\nTools push step events to an asyncio.Queue before completion.\nA background coroutine drains that queue and forwards events to the SSE generator,\nso the frontend sees "Searching → Merging → Reranking → Grading" in near-real time\n*while the tool is still running*, not just after it finishes.\n"""',
         '"""\nMedQA LangChain OpenAI-tools Agent。\n\n流式架构\n--------\nAgent 通过 `agent_executor.astream(stream_mode="messages")` 调用。\n每个 yield 出的 chunk 为 AIMessageChunk（token）或含工具调用的 ToolMessage/AIMessage。\n两者均被拦截以推送 SSE 事件。\n\n工具步骤实时可见性\n------------------\n工具在执行期间将步骤事件推入 asyncio.Queue。\n后台协程持续消费该队列并转发给 SSE 生成器，\n使前端在工具执行过程中就能实时看到\n"检索中 → 合并中 → 重排中 → 评分中"，而非等待工具结束后才显示。\n"""'),
        ('# ── Background queue drainer ──────────────────────────────────────────────',
         '# ── 后台队列消费协程 ─────────────────────────────────────────────────────────'),
        ('# The queue drainer runs concurrently with the agent stream.',
         '# 队列消费协程与 agent 流并发运行。'),
        ('# We use a sentinel value to shut it down.',
         '# 使用哨兵值关闭消费循环。'),
        ('# ── Stream agent output ───────────────────────────────────────────────────',
         '# ── 流式 agent 输出 ─────────────────────────────────────────────────────────'),
        ('# Token streaming',  '# Token 流式输出'),
        ('# Tool call start/end and intermediate steps',
         '# 工具调用开始/结束及中间步骤'),
        ('# Try to extract sources from RAG tool output',
         '# 尝试从 RAG 工具输出中提取来源'),
        ('# Final output', '# 最终输出'),
        ('# Signal queue drainer to stop', '# 通知队列消费协程停止'),
        ('# ── Merged stream via shared output queue ────────────────────────────────',
         '# ── 通过共享输出队列合并两路流 ─────────────────────────────────────────────'),
        ('# Drain any remaining tool-step events from queue',
         '# 排空队列中残余的工具步骤事件'),
        ('# Emit final sources and done', '# 推送最终来源事件和完成事件'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [agent.py] 替换: {old[:50]!r}")
        else:
            print(f"  [agent.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── api/auth.py ─────────────────────────────────────────────────────────────
def fix_api_auth():
    p = BASE / "api/auth.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""Authentication routes: register, login, profile."""',
         '"""用户认证路由：注册、登录、个人信息。"""'),
        ('# Check uniqueness', '# 检查用户名/邮箱唯一性'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [api/auth.py] 替换: {old[:50]!r}")
        else:
            print(f"  [api/auth.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── api/documents.py ────────────────────────────────────────────────────────
def fix_api_documents():
    p = BASE / "api/documents.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""Document upload and management routes.\n\nUpload flow\n-----------\n1. Validate file size / type.\n2. Create a Document row with status=processing.\n3. Extract text (PDF via pdfplumber, plain-text passthrough).\n4. Build hierarchical chunks (L1/L2/L3).\n5. Generate dense + sparse vectors for L3 chunks.\n6. Insert L3 chunks into Milvus.\n7. Persist L1/L2 parent chunks into PostgreSQL via DocStore.\n8. (Re-)fit BM25 on all L3 texts from this document.\n9. Mark document as ready.\n"""',
         '"""\n文档上传与管理路由。\n\n上传流程\n--------\n1. 校验文件大小/类型。\n2. 创建 Document 行（status=processing）。\n3. 提取文本（PDF 使用 pdfplumber；纯文本直接透传）。\n4. 构建三级分块（L1/L2/L3）。\n5. 为 L3 分块生成稠密向量 + BM25 稀疏向量。\n6. 将 L3 分块写入 Milvus。\n7. 通过 DocStore 将 L1/L2 父块持久化到 PostgreSQL。\n8. 在本文档所有 L3 文本上（重新）训练 BM25 模型。\n9. 将文档状态标记为 ready。\n"""'),
        ('"""Background task: chunk → embed → store → BM25 fit."""',
         '"""后台任务：分块 → 嵌入 → 存储 → BM25 训练。"""'),
        ('# ── 1. Chunk ──────────────────────────────────────────────────────────────',
         '# ── 1. 分块 ──────────────────────────────────────────────────────────────'),
        ('# ── 2. Dense vectors for L3 ───────────────────────────────────────────────',
         '# ── 2. L3 稠密向量 ──────────────────────────────────────────────────────'),
        ('# ── 3. BM25 fit & sparse vectors ────────────────────────────────────────',
         '# ── 3. BM25 训练 & 稀疏向量 ─────────────────────────────────────────────'),
        ('# ── 4. Milvus insert ────────────────────────────────────────────────────',
         '# ── 4. 写入 Milvus ──────────────────────────────────────────────────────'),
        ('# ── 5. PostgreSQL parent chunks ─────────────────────────────────────────',
         '# ── 5. PostgreSQL 父块持久化 ────────────────────────────────────────────'),
        ('# Delete existing document with same filename for this user (dedup)',
         '# 检测同名文档并删除旧版本（去重）'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [api/documents.py] 替换: {old[:50]!r}")
        else:
            print(f"  [api/documents.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── api/chat.py ─────────────────────────────────────────────────────────────
def fix_api_chat():
    p = BASE / "api/chat.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""Chat routes: sessions, history, and streaming SSE endpoint.\n\nSSE event types (JSON-encoded in `data:` field):\n  token       – incremental LLM output token\n  step        – RAG step trace (Searching, Merging, Reranking, Grading, Rewriting)\n  tool_start  – agent began calling a tool\n  tool_end    – agent received tool result\n  sources     – final list of retrieved/reranked sources\n  done        – generation complete, includes final_answer\n  error       – error message\n"""',
         '"""\n聊天路由：会话管理、历史记录与 SSE 流式接口。\n\nSSE 事件类型（JSON 编码于 `data:` 字段）：\n  token       – LLM 逐 token 增量输出\n  step        – RAG 步骤追踪（Searching / Merging / Reranking / Grading / Rewriting）\n  tool_start  – agent 开始调用工具\n  tool_end    – agent 收到工具结果\n  sources     – 最终检索/重排来源列表\n  done        – 生成完成，包含 final_answer\n  error       – 错误信息\n"""'),
        ('# ─── Session CRUD ─────────────────────────────────────────────────────────────',
         '# ─── 会话 CRUD ────────────────────────────────────────────────────────────────'),
        ('# ─── Streaming Chat ───────────────────────────────────────────────────────────',
         '# ─── 流式聊天 ────────────────────────────────────────────────────────────────'),
        ('"""SSE endpoint. The client passes the query as a query-param and listens for events."""',
         '"""SSE 接口。客户端以查询参数传递问题并监听事件。"""'),
        ('# Load history & possibly summarise', '# 加载历史消息并可能触发摘要'),
        ('# Save user message', '# 保存用户消息'),
        ('# invalidate so next load is fresh', '# 使缓存失效以便下次加载最新消息'),
        ('# Commit streaming message to history', '# 将流式消息落库'),
        ('# Update session title in sidebar', '# 更新侧边栏中的会话标题'),
        ('# Persist assistant message', '# 持久化助手消息'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [api/chat.py] 替换: {old[:50]!r}")
        else:
            print(f"  [api/chat.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── api/admin.py ────────────────────────────────────────────────────────────
def fix_api_admin():
    p = BASE / "api/admin.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""Admin routes: user management (admin only)."""',
         '"""管理员路由：用户管理（仅限 admin 角色）。"""'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [api/admin.py] 替换: {old[:50]!r}")
        else:
            print(f"  [api/admin.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── eval/evaluator.py ───────────────────────────────────────────────────────
def fix_evaluator():
    p = BASE / "eval/evaluator.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('"""RAG Evaluation framework.\n\nEvaluates retrieval quality across different rerank strategies.\n\nMetrics\n-------\n- Precision@k      – fraction of retrieved docs that are relevant\n- Recall@k         – fraction of relevant docs that are retrieved\n- MRR              – mean reciprocal rank of first relevant result\n- NDCG@k           – normalised discounted cumulative gain\n- Rerank lift       – improvement of rerank vs raw retrieval\n\nUsage (standalone script)\n--------------------------\n    python -m backend.eval.evaluator --config eval_config.json\n"""',
         '"""\nRAG 评估框架。\n\n对不同重排策略的检索质量进行量化评估。\n\n指标\n----\n- Precision@k  – 检索结果中相关文档的比例\n- Recall@k     – 相关文档中被检索到的比例\n- MRR          – 首个相关结果的平均倒数排名\n- NDCG@k       – 归一化折损累积增益\n- Rerank 提升  – 重排 vs 原始检索的改善量\n\n独立脚本使用方式\n----------------\n    python -m backend.eval.evaluator --config eval_config.json\n"""'),
        ('    relevant_chunk_ids: list[str]  # ground-truth relevant chunk IDs',
         '    relevant_chunk_ids: list[str]  # 标准答案相关分块 ID 列表'),
        ('"""Run evaluation for a single strategy config against all queries."""',
         '"""对单个策略配置在所有查询上运行评估。"""'),
        ('"""Evaluate all strategies and optionally save results to JSON."""',
         '"""评估所有策略，并可选将结果保存为 JSON 文件。"""'),
        ('# Retrieve', '# 召回'),
        ('# Auto-merge', '# 自动合并'),
        ('# Rerank', '# 重排序'),
        ('# Grade', '# 相关性评分'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [evaluator.py] 替换: {old[:50]!r}")
        else:
            print(f"  [evaluator.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── cache/redis_client.py ───────────────────────────────────────────────────
def fix_redis():
    p = BASE / "cache/redis_client.py"
    text = p.read_text(encoding="utf-8")

    replacements = [
        ('class RedisClient:\n    """High-level Redis helper for sessions and parent-doc caching."""',
         'class RedisClient:\n    """会话与父文档缓存的高层级 Redis 辅助类。"""'),
        ('# ── Session ──────────────────────────────────────────────────────────────',
         '# ── 会话缓存 ──────────────────────────────────────────────────────────────'),
        ('# ── Parent Document ───────────────────────────────────────────────────────',
         '# ── 父文档块缓存 ──────────────────────────────────────────────────────────'),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            print(f"  [redis_client.py] 替换: {old[:50]!r}")
        else:
            print(f"  [redis_client.py] 未找到: {old[:50]!r}")
    p.write_text(text, encoding="utf-8")

# ─── models ──────────────────────────────────────────────────────────────────
def fix_models():
    for fname, replacements in {
        "document.py": [
            ('"""Stores L1 and L2 (parent) chunks for auto-merging retrieval."""',
             '"""存储 L1/L2 父块，供自动合并检索使用。"""'),
            ('# parent_id is the L1 chunk id for a L2 chunk; NULL for L1 chunks',
             '# parent_id 为 L2 块的 L1 父块 ID；L1 块该字段为 NULL'),
        ],
    }.items():
        p = BASE / "models" / fname
        text = p.read_text(encoding="utf-8")
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                print(f"  [models/{fname}] 替换: {old[:50]!r}")
            else:
                print(f"  [models/{fname}] 未找到: {old[:50]!r}")
        p.write_text(text, encoding="utf-8")

if __name__ == "__main__":
    print("开始批量替换英文注释为中文...")
    fix_tools()
    fix_memory()
    fix_agent()
    fix_api_auth()
    fix_api_documents()
    fix_api_chat()
    fix_api_admin()
    fix_evaluator()
    fix_redis()
    fix_models()
    print("完成！")
