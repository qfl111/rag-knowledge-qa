"""
RAG 个人知识库问答 —— FastAPI 服务版
====================================
把命令行 RAG 包装成一个网页服务：浏览器打开即可提问，并能看到引用来源。

运行（任选其一）：
    方式 A（直接跑脚本）：  python app.py
    方式 B（用 uvicorn）：   uvicorn app:app --reload --port 8000

然后浏览器访问 http://127.0.0.1:8000

接口说明：
    GET  /            返回聊天界面（index.html）
    GET  /api/health  健康检查：知识库是否已建立
    POST /api/ingest  重新建立知识库（读取 ./docs 下所有文档）
    POST /api/ask     提问：body={"question":"..."} → {"answer":..., "sources":[...]}
"""

import os
import shutil

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import rag  # 复用 rag.py 里的 ingest / answer_with_sources / DB_DIR

app = FastAPI(title="RAG 个人知识库问答")

INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


class AskRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index():
    """返回聊天界面。"""
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    """简单健康检查：知识库是否已建立。"""
    ready = os.path.isdir(rag.DB_DIR) and bool(os.listdir(rag.DB_DIR))
    return {"status": "ok", "kb_ready": ready}


@app.post("/api/ingest")
def api_ingest():
    """重新建立知识库（先清空旧库，避免重复写入）。"""
    try:
        if os.path.isdir(rag.DB_DIR):
            shutil.rmtree(rag.DB_DIR)
        rag.ingest()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "done"}


@app.post("/api/ask")
def api_ask(req: AskRequest):
    """提问接口：返回答案 + 引用来源。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        answer, sources = rag.answer_with_sources(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
