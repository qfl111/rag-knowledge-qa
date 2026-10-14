# RAG 个人知识库问答系统

> 把本地文档变成「能对话的知识库」：把 PDF / Markdown / TXT 丢进 `docs/`，用中文提问，返回**带引用出处**的答案。资料里没有的内容，如实回答"资料中未提及"，不编造。

基于 **RAG（检索增强生成）** 的最小可用实现，覆盖 RAG 全链路：文档加载 → 文本切分 → 本地向量化 → 相似度检索 → 拼装 Prompt → 大模型生成（带出处标注）。

## 功能特性

- **多格式文档加载**：支持 PDF / Markdown / TXT
- **本地中文向量化**：FastEmbed + bge-small-zh-v1.5，免费、无需调用外部 Embedding API
- **持久化向量库**：Chroma 本地存储，建库一次、重复提问
- **Top-K 检索 + 引用溯源**：答案标注「来源1 / 来源2」，可回溯原文段落
- **防幻觉护栏**：检索不到相关内容时明确回答"资料中未提及"，而非胡编

## 技术栈

| 层 | 技术 |
|----|------|
| 编排 / 管线 | Python、LangChain |
| 向量化 | FastEmbed（bge-small-zh-v1.5，本地模型） |
| 向量数据库 | Chroma |
| 大模型 | DeepSeek API（OpenAI 兼容接口） |
| 配置管理 | python-dotenv |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥：复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
cp .env.example .env

# 3. 放入文档：把你的 PDF / MD / TXT 放进 ./docs 文件夹

# 4. 建库（只需运行一次；首次会自动下载中文向量模型）
python rag.py ingest

# 5. 提问
python rag.py ask 什么是RAG
```

**运行示例：**

```
$ python rag.py ask 什么是RAG

根据【来源 1】，RAG 的全称是 Retrieval-Augmented Generation（检索增强生成）……
来源: 示例文档.txt (片段 1)
```

## 目录结构

```
rag-qa/
├── rag.py                 # 全部逻辑：ingest 建库 / ask 提问
├── docs/                  # 放入你的文档（PDF / MD / TXT）
├── chroma_db/             # 向量库（ingest 后自动生成，已 gitignore）
├── requirements.txt
├── .env.example           # 密钥模板（复制为 .env 后填入真实 Key）
├── 测试集.md               # 自建 15 题评测集（测 Recall@K 用）
└── RAG学习指南与面试备战.md # 原理拆解 + 面试问答（逐行讲清代码）
```

## 评测指标

- 计算公式：`Recall@K = 召回命中数 ÷ 总题数`；`回答准确率 = 答对题数 ÷ 总题数`
- 实测结果：Recall@K=14÷15=93%  回答准确率=14÷15=93%

## 已知边界与改进方向

- 当前为「向量单路检索」，未做 BM25 混合检索与 Rerank 重排
- 切分为固定长度，未做按标题层级的语义切分
- 仅命令行交互，未提供 Web 界面（可用 Streamlit / FastAPI 补齐）
