"""修复 documents.py 后台任务的 Session 生命周期问题。"""
import re

path = "/home/zhuanz1/projects/MEDQA/backend/api/documents.py"
content = open(path).read()

# ── 1. 修复 import：补充 AsyncSessionFactory ──────────────────────────────────
old_import = "from ..database import get_db"
new_import = "from ..database import AsyncSessionFactory, get_db"
if old_import in content:
    content = content.replace(old_import, new_import)
    print("✓ import 已更新")
else:
    print("⚠ import 行未找到，跳过")

# ── 2. 修复函数签名：移除 db 参数，改为自建 Session ───────────────────────────
old_sig = "async def _process_document(doc_id: str, text: str, filename: str, db: AsyncSession) -> None:\n    \"\"\"后台任务：分块 → 嵌入 → 存储 → BM25 训练。\"\"\"\n    result = await db.execute(select(Document).where(Document.id == doc_id))    \n    doc = result.scalar_one_or_none()\n    if doc is None:\n        return"

new_sig = '''async def _process_document(doc_id: str, text: str, filename: str) -> None:
    """后台任务：分块 → 嵌入 → 存储 → BM25 训练。

    注意：后台任务必须自建 Session，不能复用请求的 Session（请求结束后 Session 已关闭）。
    """
    async with AsyncSessionFactory() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            return

        try:'''

if old_sig in content:
    # 同时把函数体里的 try 去掉（现在由外层 async with 的 try 替代）
    content = content.replace(old_sig, new_sig)
    print("✓ 函数签名已修复")
else:
    # 尝试宽松匹配（去掉尾随空格）
    old_sig2 = old_sig.replace("    \n", "\n")
    if old_sig2 in content:
        content = content.replace(old_sig2, new_sig)
        print("✓ 函数签名已修复（宽松匹配）")
    else:
        print("✗ 函数签名未匹配，打印当前内容供调试：")
        idx = content.find("async def _process_document")
        print(repr(content[idx:idx+300]))

# ── 3. 修复调用端：移除 db 参数 ──────────────────────────────────────────────
old_call = "background_tasks.add_task(_process_document, doc_id, text, file.filename or \"unknown\", db)"
new_call = "background_tasks.add_task(_process_document, doc_id, text, file.filename or \"unknown\")"
if old_call in content:
    content = content.replace(old_call, new_call)
    print("✓ 调用端已更新（移除 db 参数）")
else:
    print("⚠ 调用端未找到，请手动检查")

# ── 4. 修复函数末尾：补上对应的 except 和 async with 关闭 ────────────────────
# 原来函数末尾的 except 块需要加一层缩进，并在 async with 外补 except
old_except = '''    except Exception as e:
        logger.exception("Document processing failed for %s", doc_id)
        doc.status = "error"
        doc.error_msg = str(e)[:500]
        db.add(doc)
        await db.commit()'''

new_except = '''        except Exception as e:
            logger.exception("文档处理失败：%s", doc_id)
            try:
                doc.status = "error"
                doc.error_msg = str(e)[:500]
                db.add(doc)
                await db.commit()
            except Exception:
                pass'''

if old_except in content:
    content = content.replace(old_except, new_except)
    print("✓ except 块已更新")
else:
    print("⚠ except 块未找到，请手动检查")

open(path, "w").write(content)
print("\n完成！文件已写入。")
