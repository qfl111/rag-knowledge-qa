# RAG 个人知识库问答 · 全链路学习手册

> 从"只会学校 Python"到"独立做出一个能跑、能演示、懂原理、能标准化"的完整 RAG 问答服务。
> 四个阶段一册通：**动手搭建 → 原理深挖+面试 → FastAPI 服务化 → MCP 协议标准化**。
> 配套代码：`rag.py`（核心）、`app.py`（服务）、`index.html`（界面）、`mcp_server_demo.py`（MCP Server）。

---

# ═══════════════════════════════════════
# 阶段一：从零动手搭建（跟着做一遍就能自己搭出来）
# ═══════════════════════════════════════

## 0. 先花 30 秒懂 RAG 在干嘛

RAG = **R**etrieval（检索）+ **G**eneration（生成）。一句话：

> 你把自己的文档交给程序建库；提问时，程序**先去库里找最相关的几段**，再让大模型**只照这几段**作答，并标出来源。

为什么不直接问 DeepSeek？因为 DeepSeek 不认识你的私有资料，还爱"一本正经胡说"（幻觉）。RAG 相当于给大模型**当场发参考资料**，让它"开卷答题"。

```
你的文档(docs/) ─①加载→ 文本 ─②切分→ 多个片段(chunk)
                                          │
                                   ③向量化+存入 Chroma 向量库
                                          │
你提问 ─→ ④检索 Top-K 相关片段 ─→ 拼进 Prompt ─→ DeepSeek 作答(带出处)
```

---

## 1. 准备环境

用你本机已经装好的环境（Python 3.11 + VS Code），不用重装。

1. 新建项目文件夹，比如 `D:/project/rag-qa`，进去。
2. 打开终端（VS Code 里 `Ctrl + \`` 打开终端），确认 Python：
   ```bash
   python --version        # 应显示 3.11.x
   ```
3. 装依赖（一条命令）：
   ```bash
   pip install langchain langchain-community langchain-openai chromadb fastembed python-dotenv
   ```

---

## 2. 拿 DeepSeek API Key

1. 打开 https://platform.deepseek.com/ ，注册并充值（很便宜，几块钱够用很久）。
2. 在「API Keys」里创建一个 Key，形如 `sk-xxxx`。
3. 在项目根目录新建文件 `.env`，内容：
   ```ini
   DEEPSEEK_API_KEY=sk-你的真实key
   ```
   > 密钥**只放 `.env`，绝不写进代码、也不传 GitHub**（`.gitignore` 已挡住它）。

---

## 3. 准备你的知识库素材

新建 `docs/` 文件夹，丢几份资料进去（`.pdf` / `.md` / `.txt` 都行）。先放 3–5 份小文件，跑通后再加大的。

---

## 4. 从零写代码（rag.py）

下面按"管线顺序"一段段写。**每贴一段，先理解它干嘛，再往下。**

### 4.1 加载文档（把文件读成文本）

```python
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_docs(folder="./docs"):
    docs = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if name.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif name.endswith((".md", ".txt")):
            docs += TextLoader(path, encoding="utf-8").load()
    return docs
```
- **干嘛**：不同格式解析方式不同（PDF 要提取文字，TXT 直接读），分类型处理；`utf-8` 防止中文乱码。
- **踩坑**：扫描件 PDF 本质是图片，提取不出文字，需要 OCR——先用文字版 PDF 练手。

### 4.2 切分文本（长文切成小段）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
```
- **干嘛**：大模型一次看不了整本书，且"按小块检索"更准。`chunk_size=500` 字一段，`overlap=80` 让相邻块重叠，避免一句话被切断。

### 4.3 向量化 + 存入 Chroma（核心一步）

```python
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectordb.persist()
```
- **干嘛**：把每段文字变成一串数字（向量，512 个数）。**意思相近的文字，向量在数学空间里也"靠得近"**。
- **⚠️ 模型下载慢的坑**：`fastembed` 默认把缓存放在系统 Temp，Windows 清理后得重下。我们已把缓存挪到项目内 `model_cache/`（见 `setup_model_cache.py`），之后永不重下。

### 4.4 检索 + 拼 Prompt + 问 DeepSeek

```python
retriever = vectordb.as_retriever(search_kwargs={"k": 4})   # 每次取最相关的 4 段

llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com/v1", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "你只能根据下面的【资料】回答问题，资料里没有就如实说"资料中未提及"。\n"
    "【资料】\n{context}\n\n问题：{question}"
)
```
- **`temperature=0`**：问答要稳定准确，不让模型自由发挥。
- **"只按资料答"**：防幻觉的关键护栏——问"你好"它回"资料中未提及"，不是 bug。

### 4.5 命令行外壳

```bash
python rag.py ingest    # 建库
python rag.py ask "RAG 是怎么防幻觉的？"  # 提问
```

---

## 5. 跑通验证

**怎么算成功**：答案里能找到来自你 `docs/` 的原文意思，且问无关内容时会说"资料中未提及"。

---

## 6. 测真实指标（写进简历）

自建 15 条测试集，统计 Recall@K 和回答准确率。本项目实测：**Recall@K ≈ 93%，回答准确率 ≈ 93%**（详见 `测试集.md`）。

---

## 7. 常见坑

| 现象 | 原因 | 解决 |
|------|------|------|
| `ingest` 卡在下载模型 | fastembed 走限速镜像 + Temp 缓存被清 | 用项目内 `model_cache/`，或换 ModelScope 源 |
| 回答"资料中未提及" | 防幻觉护栏生效 | 确认 docs 里真有相关内容 |
| 中文乱码 | 编码非 utf-8 | TextLoader 显式 `encoding="utf-8"` |
| 问 PDF 没内容 | 扫描件 PDF 是图片 | 先换文字版 PDF |

---

# ═══════════════════════════════════════
# 阶段二：原理深挖 + 面试备战
# ═══════════════════════════════════════

## 核心名词速查表

| 名词 | 一句话解释 |
|------|-----------|
| **LLM / 大模型** | 像 DeepSeek 这种"读了很多书、能写能答"的 AI 模型 |
| **API** | 你发一段请求、它返回答复的"网络接口"，调用要 Key + 花钱(很少) |
| **Token** | 大模型计费/计长的最小单位，中文约 1–2 字=1 token |
| **Embedding 向量** | 把文字变成一串数字，意思相近的文字数字也相近 |
| **向量数据库** | 专门存向量、能快速找"最近邻居"的数据库（本项目的 Chroma） |
| **RAG** | 检索增强生成：先检索资料，再让大模型"照资料答" |
| **幻觉** | 大模型编假内容；RAG 的"只按资料答"就是用来防它 |
| **chunk / 分块** | 把长文切成小段，便于精确检索 |
| **Top-K** | 检索时返回最相关的 K 个结果（本项目 k=4） |

---

## 逐环节拆解

### 环节 1：加载文档（`load_docs`）
- **是什么**：把各种格式的文档读成"纯文本 + 元数据"。
- **为什么**：不同文件格式解析方式不同；`utf-8` 防中文乱码。
- **学什么**：文件 IO、编码、PDF 文本提取的坑（扫描件是图片，需 OCR）。

### 环节 2：切分文本（RecursiveCharacterTextSplitter）
- **是什么**：把长文切成很多 500 字左右的小段，段与段重叠 80 字。
- **为什么**：大模型上下文有限，按小块检索更准；重叠避免语义被切断。
- **学什么**：分块策略；`chunk_size` 太大→检索不精，太小→语义碎。

### 环节 3：向量化 / Embedding（FastEmbed + bge-small-zh-v1.5）
- **是什么**：把中文片段变成一串数字向量（512 维）。意思相近的句子向量距离近。
- **为什么用本地模型**：免费、不上传文档、快、无调用延迟上限。
- **学什么**：向量、余弦相似度、embedding 模型是预训练好的。

### 环节 4：向量数据库 Chroma
- **是什么**：专门存"向量 + 原文"的数据库，支持"给我一个问题向量，返回最像的几个片段"。
- **为什么**：自己算所有距离太慢；向量库专门优化了这个查询。

### 环节 5：检索 Top-K
- **为什么 k=4**：太少漏答案，太多塞噪声、费 token。4 是平衡值。

### 环节 6：拼 Prompt 交给 DeepSeek
- **`temperature=0`**：问答场景要稳定准确。
- **"只按资料答"**：防幻觉护栏。问"你好"回"资料中未提及"=护栏生效。
- **DeepSeek 为什么用 `ChatOpenAI`**：兼容 OpenAI 接口格式，换 `base_url` 即可。

---

## 面试问答（HR + 技术面）

### HR / 行为类

**Q：你为什么做这个项目？**
> 想投 AI Agent 方向，但零基础没实习。RAG 是 Agent 底层能力之一，我就自己从环境搭起做了这个项目来真正理解它。

**Q：遇到最大的困难？**
> 中文向量模型下载，国内直连 HuggingFace 被墙、镜像源文件损坏。我换阿里 ModelScope 国内源用 curl 拉取正版文件解决。

**Q：学到了什么？**
> 不只是会调 API。我搞懂了 RAG 防幻觉原理、embedding 怎么把文字变向量、为什么检索要切分和取 Top-K。也第一次完整走通了"装环境→配密钥→跑工程→排查报错"的流程。

### 技术类（重点准备）

**Q：RAG 的原理？为什么不直接问大模型？**
> RAG = 检索 + 生成。先检索相关片段，再拼进 Prompt 让大模型作答。直接问大模型不知道私有文档，且会幻觉。RAG 让回答"有依据、可溯源"。

**Q：什么是 Embedding？**
> 把文字映射成数字向量，语义相近的向量距离近。检索时问题也向量化，算相似度取最近的几个。本项目用本地中文模型 `bge-small-zh-v1.5`，不联网、免费。

**Q：为什么要切分文档？chunk_size 怎么定？**
> 大模型上下文有限，按小块检索更精确。chunk 太大检索不精准，太小语义碎。我用 500 字 + 80 字重叠。

**Q：怎么避免幻觉？**
> 两点：① Prompt 要求"只根据资料回答，没有就如实说未提及"；② 保证检索到的资料确实相关。

**Q：怎么评估 RAG 效果？**
> 建问答测试集，看检索召回率（Recall@K）和回答准确率。我自建 15 条测试集实测 Recall@K ≈ 93%。

**Q：向量模型为什么用本地？**
> 文档不出本机隐私好；免费无延迟上限；不依赖额外密钥。

**Q：DeepSeek 为什么用 `ChatOpenAI`？**
> 兼容 OpenAI 接口格式，换 `base_url` 就能用。

**Q：几万篇文档还行吗？**
> 当前单库简单检索，规模大了要：① 更好的分块和索引；② 混合检索（向量+BM25）；③ 加 Reranker；④ 可能换 Milvus。

---

# ═══════════════════════════════════════
# 阶段三：FastAPI 服务化（脚本→产品）
# ═══════════════════════════════════════

## 为什么需要"服务化"

命令行版本只有会敲命令的人能用。服务化解决三件事：
1. **有界面**：浏览器打开就能问；
2. **有接口**：任何程序都能 HTTP 调用你的 RAG；
3. **可部署**：改一改就能上线。

**脚本 → 服务 → 产品** 的跃迁，是简历里很加分的"工程化"证据。

---

## 整体架构

```
浏览器 (index.html)
      │  HTTP POST /api/ask {"question":"..."}
      ▼
FastAPI 服务 (app.py)
      │  调用
      ▼
RAG 核心 (rag.py: answer_with_sources)
      │  检索 + 大模型
      ▼
答案 + 引用来源  ──►  返回 JSON  ──►  前端渲染成对话气泡
```

**前端不懂 RAG，后端不懂界面**，通过 HTTP + JSON 解耦——这正是真实工程的分工方式。

---

## FastAPI 核心概念

| 概念 | 对应代码 | 一句话 |
|------|---------|--------|
| 应用实例 | `app = FastAPI()` | "网站服务"本体 |
| 路由 | `@app.post("/api/ask")` | 把 URL 绑定到函数 |
| 请求体 | `class AskRequest(BaseModel)` | 自动校验 + 解析 JSON |
| 返回 | `return {"answer":..., "sources":[...]}` | 自动序列化 JSON |
| 运行 | `uvicorn.run(app, port=8000)` | 把服务跑起来 |

---

## 怎么跑

```bash
cd 项目/rag-qa
python app.py
# 浏览器打开 http://127.0.0.1:8000
```

---

## 面试话术

> "我把 RAG 核心封装成了 FastAPI 服务：`@app.post('/api/ask')` 暴露问答接口，请求体用 Pydantic 模型校验，返回 JSON（答案 + 引用来源）；前端是原生 JS 单页聊天界面。相比纯脚本，它获得了**可交互界面、标准 HTTP 接口、可部署能力**——从 Demo 走向工程化。"

---

## 可扩展方向

1. **流式输出（Streaming）**：答案逐字吐出；
2. **鉴权**：加 token 防止乱调 DeepSeek 额度；
3. **部署上线**：Docker + 云服务器，变成公网可访问的应用。

---

# ═══════════════════════════════════════
# 阶段四：MCP 协议标准化（能力→插件）
# ═══════════════════════════════════════

## MCP 解决的核心痛点：M × N 集成地狱

- 没有 MCP：N 个工具 × M 个客户端 = **M×N** 套对接代码
- 有了 MCP：每个工具实现一次 Server + 每个客户端实现一次 Client = **M+N** 套代码

**MCP 把"工具↔大模型应用"从两两硬编码，变成插拔式标准接口**（类比 USB 接口统一了外设）。

---

## 三个角色

| 角色 | 是什么 | 类比 | 本项目 |
|------|--------|------|--------|
| **Host** | 运行大模型的应用 | 你的电脑 | Claude 桌面端、Cursor |
| **MCP Client** | Host 里负责连 Server 的部件 | USB 口 | Host 内置 |
| **MCP Server** | 你写的、暴露工具的程序 | USB 设备 | `mcp_server_demo.py` |

---

## 和你写过的代码对照

### 之前手写工具（file-agent）
```python
@tool
def move_file(src: str, dst: str) -> str: ...
TOOLS = [list_dir, read_head, move_file]
llm = llm.bind_tools(TOOLS)
```

### 现在用 MCP
```python
from fastmcp import FastMCP
mcp = FastMCP("rag-knowledge-qa-mcp")

@mcp.tool()
def rag_query(question: str) -> str:
    """基于个人知识库回答..."""
    answer, sources = rag.answer_with_sources(question)
    return f"{answer}\n\n--- 引用来源 ---\n..."
```

**区别**：`@tool` → `@mcp.tool()`，写法几乎一样，只是挂到 MCP 上；`bind_tools` + 手写循环 → MCP Client 负责。你的 RAG 函数原封不动复用。

> **MCP 不取代你的 Agent 逻辑，它取代的是"每个客户端都要单独写一遍工具对接"这件事。**

---

## 怎么接到真实客户端

在 Claude 桌面端 / Cursor 的 MCP 配置文件里加：

```json
{
  "mcpServers": {
    "rag-knowledge-qa": {
      "command": "D:/Python311/python.exe",
      "args": ["D:/Work Buddy_files/简历制作/项目/rag-qa/mcp_server_demo.py"]
    }
  }
}
```

配置好后，你的 RAG 瞬间变成任何 MCP 客户端的"插件"。

---

## 面试话术

> "MCP 是 Anthropic 提出的开放协议，用来标准化大模型应用连接外部工具与数据源的方式，解决 M×N 集成问题。我之前手写 Function Calling 工具，后来把 RAG 问答封装成 MCP Server（用 `@mcp.tool` 暴露 `rag_query`），任何支持 MCP 的客户端都能即插即用。"

---

## 常见误区

- ❌ "MCP 是新的 Agent 框架" → 错，它是**协议/接口标准**。
- ❌ "用了 MCP 就不用写工具函数了" → 错，工具函数照样写，只是改用 `@mcp.tool()` 暴露。
- ❌ "MCP 只服务于 Claude" → 错，开放协议，任何客户端都能用。

---

# ═══════════════════════════════════════
# 全链路故事线（面试终极话术）
# ═══════════════════════════════════════

你这一路把 AI Agent 工程化的主干全摸了一遍：

1. **RAG**（检索增强）→ 让模型"有知识"
2. **Agent + 工具调用**（file-agent / LangGraph）→ 让模型"能干活"
3. **FastAPI 服务化** → 让能力"能被调用、能演示"
4. **MCP 协议** → 让能力"标准化、能被任意客户端即插即用"

面试一句话：

> **"我从底层数据检索，到 Agent 编排，到服务暴露，再到协议标准化，全链路都动手实现过。"**

这比单纯说"我会 LangChain"有说服力得多。

---

*本手册合并自：RAG从零动手教程.md、RAG学习指南与面试备战.md、FastAPI服务化学习资料.md、MCP概念学习资料.md*
