"""
MCP 演示服务器（Demo）
=====================
把"你已实现的 RAG 问答"和"文件查看"封装成标准 MCP 工具。
这样任何支持 MCP 的客户端（Claude 桌面端 / Cursor / 你自己的 Agent）
都能像调用原生能力一样调用你的知识库，而无需为每个客户端写定制代码。

运行方式（作为 MCP server 被客户端拉起，不要手动 python 运行后交互）：
    把它配置进客户端的 mcpServers（见 MCP概念学习资料.md 的 JSON 配置），
    客户端启动时会自动用 stdio 拉起本进程。

本文件还可在普通 Python 里直接 import 并调用 `rag_query(...)` 做端到端验证。

依赖：pip install fastmcp
"""
import os
import sys

# 让本文件能 import 同目录的 rag.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag  # 复用 RAG 核心：answer_with_sources

from fastmcp import FastMCP

# 创建一个 MCP server 实例，名字随意，会显示在客户端里
mcp = FastMCP("rag-knowledge-qa-mcp")


@mcp.tool()
def rag_query(question: str) -> str:
    """基于个人知识库回答用户问题。仅依据本地文档，返回带引用的答案。

    Args:
        question: 用户的中文问题，例如"什么是 AI Agent？"
    """
    answer, sources = rag.answer_with_sources(question)
    src_text = "\n\n".join(f"[{i + 1}] {s[:200]}" for i, s in enumerate(sources))
    return f"{answer}\n\n--- 引用来源 ---\n{src_text}"


@mcp.tool()
def list_inbox(directory: str = "./docs") -> str:
    """列出指定目录下的文件，用于文件整理类 Agent 感知环境。

    Args:
        directory: 要查看的目录路径，默认 ./docs
    """
    try:
        files = sorted(os.listdir(directory))
        return "\n".join(files) if files else "(空目录)"
    except Exception as e:
        return f"错误: {e}"


if __name__ == "__main__":
    # 默认使用 stdio 传输：客户端通过标准输入输出与本进程通信
    mcp.run()
