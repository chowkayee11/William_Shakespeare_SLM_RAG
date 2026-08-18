# Shakespeare SLM/RAG System — CSCI433/933 Assignment 2

A domain-adapted Small Language Model (SLM) system with Retrieval-Augmented
Generation (RAG) for answering questions about Shakespeare's plays:
**Hamlet**, **Macbeth**, and **Romeo and Juliet**.

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Embedding Model (all-MiniLM-L6-v2)        │
│  → encode query into 384-dim vector         │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Retriever (cosine similarity)              │
│  → find top-k most relevant scene chunks    │
│  → supports dense / sparse / hybrid modes   │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Prompt Builder                             │
│  → system prompt + retrieved context +      │
│    user question → structured RAG prompt    │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  SLM (TinyLlama-1.1B-Chat / configurable)  │
│  → generate grounded, beginner-friendly     │
│    answer citing act/scene references       │
└─────────────┬───────────────────────────────┘
              │
              ▼
         Answer + Evidence
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up dataset

```bash
python setup_data.py
```

This copies the Shakespeare dataset files into `data/processed/`.

### 3. Build the retrieval index

```bash
python src/build_index.py
```

This loads the dataset, creates scene-level chunks, generates embeddings,
and saves the index for fast reuse.

### 4. Run the interactive chatbot

```bash
python src/rag_chatbot.py
```

### 5. Run the evaluation pipeline

```bash
# Instructor questions only
python src/evaluate.py

# Include group-designed questions
python src/evaluate.py --include-group
```

Results are saved to `results/evaluation_results.csv` and `.json`.

## LLM Backend Configuration

The system supports three LLM backends, configured via environment variables:

### HuggingFace (default — local SLM)

```bash
set HF_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set LLM_BACKEND=huggingface
python src/rag_chatbot.py
```

### Ollama (local server)

```bash
set LLM_BACKEND=ollama
set OLLAMA_MODEL=tinyllama
python src/rag_chatbot.py
```

### OpenAI-compatible API

```bash
set LLM_BACKEND=openai_compatible
set OPENAI_API_BASE=http://localhost:1234/v1
set OPENAI_MODEL=local-model
python src/rag_chatbot.py
```

## Project Structure

```
├── src/
│   ├── config.py          # Centralised configuration
│   ├── data_loader.py     # Dataset loading (JSON + JSONL)
│   ├── chunking.py        # Scene-level chunking with enrichment
│   ├── retrieval.py       # Dense/sparse/hybrid retrieval
│   ├── llm_interface.py   # Multi-backend LLM abstraction
│   ├── baseline.py        # Baseline system (no retrieval)
│   ├── rag_chatbot.py     # RAG pipeline + interactive chat
│   ├── evaluate.py        # Evaluation pipeline
│   └── build_index.py     # Index build + sanity check
├── data/
│   ├── processed/         # Play JSON/JSONL files
│   └── index/             # Saved retrieval index
├── prompts/
│   └── system_prompt.txt  # System instruction for the SLM
├── results/               # Evaluation outputs
├── report/                # Assignment report
├── setup_data.py          # One-time data setup script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Design Decisions

### Chunking Strategy: Scene-Level

We use **scene-level chunks** as the primary retrieval unit because:
- Each scene is a self-contained dramatic episode
- Scene text length (200–800 tokens) fits within SLM context windows
- Instructor-provided summaries and keywords improve retrieval accuracy
- Conversational flow and stage directions are preserved

### Embedding Model: all-MiniLM-L6-v2

Chosen for its balance of quality and efficiency:
- 384-dimensional embeddings
- ~80 MB model size
- Strong performance on semantic similarity benchmarks
- Fast encoding (~5,000 sentences/second on GPU)

### SLM: TinyLlama-1.1B-Chat

Selected as the default local model because:
- 1.1B parameters — fits in ≤4 GB VRAM
- Chat-finetuned with ChatML template
- Apache-2.0 licence (no restrictions)
- Adequate quality for grounded QA tasks

### Retrieval: Dense + Optional Hybrid

- **Dense mode** (default): cosine similarity on sentence embeddings
- **Hybrid mode**: weighted combination of dense + TF-IDF sparse scores
- Hybrid mode improves recall for keyword-heavy queries (character names, locations)

## Evaluation

The evaluation pipeline scores responses on 5 criteria (1-5 scale):

| Criterion | Description |
|-----------|-------------|
| Correctness | Is the answer factually accurate? |
| Grounding | Is the answer supported by retrieved evidence? |
| Retrieval Relevance | Are the retrieved passages relevant? |
| Usefulness | Would a beginner find this helpful? |
| Style Quality | Is the language clear and appropriate? |

Both baseline (no retrieval) and RAG responses are generated for each
question, enabling direct comparison.
