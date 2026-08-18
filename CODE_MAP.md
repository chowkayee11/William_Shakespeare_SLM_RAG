# Shakespeare SLM/RAG System - Code Map

## 项目结构
```
William_Shakespeare_SLM_RAG/
├── src/                      # 源代码目录
├── data/                     # 数据目录
│   ├── processed/            # 处理后的数据
│   └── index/                # 检索索引
├── prompts/                  # 提示词
├── report/                   # 报告
└── results/                  # 结果
```

---

## 代码模块详解

### 1. config.py - 中央配置模块

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `PROJECT_ROOT` | Path | 项目根目录 |
| `DATA_DIR` | Path | 处理数据目录 |
| `PLAY_NAMES` | list | 剧本列表: ["hamlet", "macbeth", "romeo_and_juliet"] |
| `DEFAULT_TOP_K` | int | 默认检索结果数: 5 |
| `EMBEDDING_MODEL_NAME` | str | 嵌入模型: "sentence-transformers/all-MiniLM-L6-v2" |
| `LLM_BACKEND` | str | LLM后端: "huggingface" / "ollama" / "openai_compatible" |
| `HF_MODEL_NAME` | str | HuggingFace模型: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" |
| `MAX_NEW_TOKENS` | int | 最大生成词元: 512 |
| `TEMPERATURE` | float | 采样温度: 0.7 |
| `USE_SCENE_JSONL` | bool | 是否使用预构建的场景JSONL: True |
| `ENRICH_CHUNK_TEXT` | bool | 是否丰富分块文本: True |

---

### 2. data_loader.py - 数据加载模块

| 函数 | 功能 | 返回值 |
|------|------|--------|
| `load_play_json(path)` | 加载完整剧本JSON文件 | `Dict[str, Any]` |
| `load_scene_chunks_jsonl(path)` | 加载场景级JSONL文件 | `List[Record]` |
| `load_all_scene_chunks()` | 加载所有三个剧本的场景分块 | `List[Record]` |
| `load_all_plays_scenes()` | 从完整JSON加载所有场景 | `List[Record]` |
| `load_instructor_questions()` | 加载教师提供的评估问题 | `List[Dict[str, str]]` |
| `_extract_scenes(obj)` | 从JSON对象中提取场景（内部函数） | `List[Record]` |

**使用示例**:
```python
from data_loader import load_all_scene_chunks
chunks = load_all_scene_chunks()  # 返回73个场景分块
```

---

### 3. chunking.py - 分块模块

| 函数 | 功能 | 返回值 |
|------|------|--------|
| `create_chunks(records)` | 将场景记录转换为检索分块 | `List[Chunk]` |
| `_build_enriched_text(record)` | 构建丰富文本（元数据+场景） | `str` |
| `_get_raw_text(record)` | 提取原始文本 | `str` |
| `format_chunk_for_display(chunk, include_text)` | 格式化分块用于显示 | `str` |
| `format_chunk_for_prompt(chunk, max_chars)` | 格式化分块用于LLM提示 | `str` |

**分块结构**:
```python
Chunk = {
    "chunk_id": str,           # 唯一标识
    "play": str,               # 剧本名
    "act": str,                # 幕
    "scene": str,              # 场
    "location": str,           # 地点
    "scene_summary": str,      # 场景摘要
    "keywords": list,          # 关键词
    "text": str,               # 用于嵌入的文本（丰富或原始）
    "raw_text": str,           # 原始场景文本
    "metadata": dict           # 完整原始记录
}
```

**元数据丰富格式**:
```
Play: Macbeth, Act 1, Scene 7 — Macbeth's castle
Summary: Macbeth debates whether to murder Duncan.
Keywords: ambition, murder, conscience
[原始场景文本...]
```

---

### 4. retrieval.py - 检索模块

| 类/函数 | 功能 |
|---------|------|
| **`EmbeddingRetriever`** | 检索器主类 |
| `.build_index(chunks)` | 构建索引（嵌入+TF-IDF） |
| `.retrieve(query, top_k)` | 检索Top-k相关分块 |
| `.save_index(path)` | 保存索引到磁盘 |
| `.load_index(path)` | 从磁盘加载索引 |

**检索模式**:
- `dense`: 嵌入余弦相似度（默认）
- `sparse`: TF-IDF余弦相似度
- `hybrid`: 加权组合: `score = α·s_dense + (1-α)·s_sparse` (α=0.7)

**使用示例**:
```python
from retrieval import EmbeddingRetriever
from config import EMBEDDING_MODEL_NAME

retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME, mode="dense")
retriever.build_index(chunks)
results = retriever.retrieve("Why does Macbeth kill Duncan?", top_k=5)
# 返回 [(chunk, score), ...]
```

---

### 5. llm_interface.py - LLM接口模块

| 函数 | 功能 | 后端 |
|------|------|------|
| `generate(prompt, system_prompt, backend)` | 统一生成接口 | 任意 |
| `_generate_hf(prompt, system_prompt)` | HuggingFace本地生成 | huggingface |
| `_generate_ollama(prompt, system_prompt)` | Ollama服务器生成 | ollama |
| `_generate_openai(prompt, system_prompt)` | OpenAI兼容API生成 | openai_compatible |
| `_load_hf_pipeline()` | 加载HuggingFace pipeline（单例） | - |

**环境变量配置**:
```bash
# HuggingFace (默认)
set LLM_BACKEND=huggingface
set HF_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Ollama
set LLM_BACKEND=ollama
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_MODEL=tinyllama

# OpenAI兼容
set LLM_BACKEND=openai_compatible
set OPENAI_API_BASE=http://localhost:1234/v1
set OPENAI_MODEL=local-model
```

**使用示例**:
```python
from llm_interface import generate
answer = generate(
    prompt="Who is Hamlet?",
    system_prompt="You are a Shakespeare expert."
)
```

---

### 6. rag_chatbot.py - RAG主管道

| 函数 | 功能 |
|------|------|
| `rag_answer(query, retriever, top_k)` | 生成RAG答案 |
| `build_rag_prompt(query, retrieved, max_context_chars)` | 构建RAG提示 |
| `load_system_prompt()` | 加载系统提示词 |
| `build_rag_system(mode, use_saved_index)` | 构建/加载完整RAG系统 |
| `interactive_chat()` | 交互式聊天循环 |

**RAG流程**:
```
用户查询
    ↓
查询嵌入 (all-MiniLM-L6-v2)
    ↓
Top-k检索 (余弦相似度)
    ↓
构建提示: [系统提示] + [检索上下文] + [用户问题]
    ↓
SLM生成答案 (TinyLlama-1.1B)
    ↓
返回答案 + 检索证据
```

**使用示例**:
```python
from rag_chatbot import build_rag_system, rag_answer

retriever = build_rag_system(mode="dense", use_saved_index=True)
answer, retrieved = rag_answer("Why does Macbeth kill Duncan?", retriever)
```

---

### 7. build_index.py - 索引构建脚本

```bash
python src/build_index.py
```
功能:
- 加载场景分块
- 创建分块
- 构建索引
- 保存索引到 `data/index/retrieval_index.pkl`

---

### 8. evaluate.py - 评估脚本

```bash
python src/evaluate.py              # 仅教师问题
python src/evaluate.py --include-group  # 包含小组问题
```

输出:
- `results/evaluation_results.csv`
- `results/evaluation_results.json`

评估标准 (1-5分):
1. Correctness - 事实准确性
2. Grounding - 检索证据支持度
3. Retrieval Relevance - 检索相关性
4. Usefulness - 对初学者的价值
5. Style Quality - 语言质量

---

### 9. baseline.py - 基线系统

| 函数 | 功能 |
|------|------|
| `baseline_answer(query)` | 无检索的基线答案 |

基线系统直接将问题发送给SLM，不使用任何检索上下文，用于对比实验。

---

## 关键数据流程

### 离线索引流程
```
1. setup_data.py → 复制数据到 data/processed/
   ↓
2. load_all_scene_chunks() → 加载73个场景
   ↓
3. create_chunks() → 创建分块 + 元数据丰富
   ↓
4. build_index() → 
   - 稠密嵌入 (all-MiniLM-L6-v2)
   - TF-IDF向量化 (ngram 1-2)
   ↓
5. save_index() → 保存到 data/index/
```

### 在线推理流程
```
用户查询
   ↓
retrieve() → 
   - 查询编码
   - 余弦相似度计算
   - Top-k排序
   ↓
build_rag_prompt() →
   - 系统提示
   - 格式化的检索上下文
   - 用户问题
   ↓
generate() → SLM生成
   ↓
返回答案 + 检索证据
```

---

## 快速参考索引

| 功能 | 文件位置 |
|------|---------|
| 元数据丰富 | `chunking.py:41-75` |
| 稠密嵌入 | `retrieval.py:104-108` |
| TF-IDF | `retrieval.py:112-120` |
| 混合检索 | `retrieval.py:134-146` |
| HuggingFace后端 | `llm_interface.py:85-122` |
| Ollama后端 | `llm_interface.py:127-154` |
| OpenAI后端 | `llm_interface.py:159-191` |
| RAG提示构建 | `rag_chatbot.py:49-85` |
| 交互式聊天 | `rag_chatbot.py:159-189` |
