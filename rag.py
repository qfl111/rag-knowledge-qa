"""
RAG 个人知识库问答 —— 最小可跑版本（约 60 行核心代码）
========================================================
能干嘛：把 ./docs 里的 PDF / Markdown / TXT 文档向量化建库，
        然后你用中文提问，它返回「带引用出处」的答案。

跑起来（5 步）：
  1) pip install -r requirements.txt
  2) 复制 .env.example 为 .env，填入你的 DEEPSEEK_API_KEY
  3) 把文档放进 ./docs 文件夹（可放多个）
  4) python rag.py ingest      # 建知识库（只需一次）
  5) python rag.py ask 什么是注意力机制   # 提问

依赖说明：向量化用 FastEmbed 本地小模型（bge-small-zh），
          不依赖联网大模型，速度快、免费；问答用 DeepSeek API。
"""

import os
import glob
import argparse

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")

DOCS_DIR = "./docs"          # 放文档的文件夹
DB_DIR = "./chroma_db"       # 向量库保存位置
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
# 向量模型永久缓存目录：放在项目内，避免系统 Temp 被清理后反复重新下载
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")


def load_docs():
    """读取 ./docs 下所有 PDF / MD / TXT 为 LangChain Document 列表。"""
    docs = []
    for path in glob.glob(f"{DOCS_DIR}/**/*", recursive=True):
        low = path.lower()
        if low.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif low.endswith((".md", ".txt")):
            docs += TextLoader(path, encoding="utf-8").load()
    return docs


def ingest():
    """切分文档 → 向量化 → 存入 Chroma。"""
    print("读取文档中…")
    raw = load_docs()
    if not raw:
        print(f"⚠️ 在 {DOCS_DIR} 没找到文档，请先放入 PDF/MD/TXT 再试。")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(raw)
    print(f"切分完成：共 {len(chunks)} 个文本片段")

    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    vectordb.persist()
    print("✅ 知识库已建立，可以开始提问了。")


def format_docs(docs):
    """把检索到的片段拼成带编号的上下文。"""
    return "\n\n".join(f"[来源 {i+1}] {d.page_content}" for i, d in enumerate(docs))


def build_qa_chain():
    """构建 RAG 问答链（检索 → 拼 Prompt → 大模型 → 解析）。"""
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE)
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=API_KEY,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        "你是一个严谨的问答助手。只根据下面的【资料】回答；"
        "如果资料里没有答案，就如实说「资料中未提及」。\n"
        "【资料】\n{context}\n\n【问题】{question}\n\n"
        "回答时尽量标注出处（如「来源1」「来源2」）："
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_with_sources(question: str):
    """返回 (答案文本, 引用片段列表)，供 Web / API 调用。"""
    chain = build_qa_chain()
    answer = chain.invoke(question)
    # 单独再检索一次，拿到用于前端展示的引用原文
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE)
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    docs = vectordb.as_retriever(search_kwargs={"k": 4}).invoke(question)
    return answer, [d.page_content for d in docs]


def ask(question: str):
    """命令行版本：直接打印答案。"""
    answer, _ = answer_with_sources(question)
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 个人知识库问答")
    parser.add_argument("cmd", choices=["ingest", "ask"], help="ingest=建库, ask=提问")
    parser.add_argument("text", nargs="*", help="提问内容（ask 时使用）")
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest()
    else:
        question = " ".join(args.text)
        if not question:
            print("用法：python rag.py ask 你的问题")
        else:
            ask(question)
