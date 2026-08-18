# Shakespeare RAG Chatbot

A local-first Retrieval-Augmented Generation (RAG) chatbot for answering questions about Shakespeare's **Hamlet**, **Macbeth**, and **Romeo and Juliet**.

The system retrieves relevant scene-level evidence from the plays, builds a grounded prompt, and uses a configurable small language model backend to generate beginner-friendly answers with act and scene references.

## Features

- Scene-level retrieval over Shakespeare play data
- Dense semantic search with `sentence-transformers/all-MiniLM-L6-v2`
- Optional sparse and hybrid retrieval support
- Configurable LLM backends:
  - HuggingFace local models
  - Ollama local models
  - OpenAI-compatible APIs
- Evidence-aware answer generation
- Baseline vs RAG evaluation pipeline
- Reusable modular Python code for loading, chunking, retrieval, prompting, generation, and evaluation

## Tech Stack

- Python
- NumPy
- pandas
- scikit-learn
- sentence-transformers
- PyTorch
- HuggingFace Transformers
- Ollama
- Requests

## Architecture

```text
User Query
    |
    v
Embedding Model
sentence-transformers/all-MiniLM-L6-v2
    |
    v
Retriever
Dense cosine similarity / sparse TF-IDF / hybrid retrieval
    |
    v
Top-k Scene Evidence
Relevant passages with play, act, and scene metadata
    |
    v
Prompt Builder
System instruction + retrieved evidence + user question
    |
    v
LLM Backend
HuggingFace / Ollama / OpenAI-compatible API
    |
    v
Grounded Answer
Answer with supporting evidence references
```

## Project Structure

```text
.
├── src/
│   ├── config.py          # Paths, model settings, retrieval settings
│   ├── data_loader.py     # Dataset loading utilities
│   ├── chunking.py        # Scene and utterance-window chunking
│   ├── retrieval.py       # Dense, sparse, and hybrid retrieval
│   ├── llm_interface.py   # LLM backend abstraction
│   ├── baseline.py        # Non-RAG baseline generation
│   ├── rag_chatbot.py     # Interactive RAG chatbot
│   ├── evaluate.py        # Evaluation pipeline
│   └── build_index.py     # Retrieval index builder
├── data/
│   ├── processed/         # Processed Shakespeare play data
│   └── index/             # Generated retrieval index
├── prompts/
│   └── system_prompt.txt  # System prompt for answer generation
├── results/               # Evaluation outputs
├── setup_data.py          # Dataset setup helper
├── requirements.txt       # Python dependencies
└── README.md
```

## Quick Start

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you run into NumPy compatibility issues with PyTorch, use NumPy 1.x:

```bash
pip install "numpy<2" --force-reinstall
```

### 3. Prepare the dataset

```bash
python setup_data.py
```

### 4. Build the retrieval index

```bash
python src/build_index.py
```

This loads the processed Shakespeare data, creates retrieval chunks, generates embeddings, and saves the retrieval index under `data/index/`.

### 5. Run the chatbot

```bash
python src/rag_chatbot.py
```

Example question:

```text
Who is Hamlet?
```

Exit the chatbot with:

```text
quit
```

## Using Ollama

Ollama is the recommended local backend for quick testing.

Start the Ollama server in one terminal:

```bash
ollama serve
```

In another terminal, run:

```bash
cd path/to/William_Shakespeare_SLM_RAG
source .venv/bin/activate

ollama pull tinyllama

export LLM_BACKEND=ollama
export OLLAMA_MODEL=tinyllama

python src/rag_chatbot.py
```

## Using HuggingFace

The default backend is HuggingFace. It downloads and runs a local model through `transformers`.

```bash
export LLM_BACKEND=huggingface
export HF_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

python src/rag_chatbot.py
```

## Using an OpenAI-Compatible API

You can also connect the chatbot to any OpenAI-compatible local or hosted API.

```bash
export LLM_BACKEND=openai_compatible
export OPENAI_API_BASE=http://localhost:1234/v1
export OPENAI_API_KEY=not-needed
export OPENAI_MODEL=local-model

python src/rag_chatbot.py
```

## Configuration

Core settings are centralized in `src/config.py`.

Useful options include:

```python
DEFAULT_TOP_K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_BACKEND = "huggingface"
HF_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OLLAMA_MODEL = "tinyllama"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
CHUNK_STRATEGY = "scene"
```

For faster or more deterministic local responses, lowering generation length and temperature can help:

```python
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.2
```

## Evaluation

Run the evaluation pipeline:

```bash
python src/evaluate.py
```

The evaluation compares baseline generation against RAG-based generation and saves outputs to:

```text
results/evaluation_results.csv
results/evaluation_results.json
```

The scoring dimensions include:

| Criterion | Description |
| --- | --- |
| Correctness | Whether the answer is factually accurate |
| Grounding | Whether the answer is supported by retrieved evidence |
| Retrieval Relevance | Whether retrieved passages are relevant |
| Usefulness | Whether the answer is helpful to a beginner |
| Style Quality | Whether the response is clear and readable |

## Retrieval Design

The default retrieval unit is a scene-level chunk. This works well for Shakespeare QA because each scene usually contains a coherent dramatic event, character interaction, or plot development.

The retriever supports:

- Dense retrieval using sentence-transformer embeddings
- Sparse retrieval using TF-IDF
- Hybrid retrieval combining dense and sparse scores

Scene metadata such as play title, act, scene, summary, and keywords is included in chunk text to improve retrieval quality for both semantic and keyword-based questions.

## Example Output

```text
You: Why does Hamlet delay his revenge?

--- Retrieved Evidence ---

[Rank 1 | Score: 0.6489]
[Hamlet, Act 3, Scene 1]
  Summary: Hamlet reflects on existence, rejects Ophelia, and Claudius suspects danger.

[Rank 2 | Score: 0.6339]
[Hamlet, Act 2, Scene 2]
  Summary: Rosencrantz and Guildenstern arrive; Hamlet plans to use players to test Claudius.

--- Answer ---
Hamlet delays his revenge because he wants stronger proof that Claudius is guilty. In Act 2, Scene 2, he decides to use the visiting players to stage a performance that mirrors the murder and observe Claudius's reaction. This shows that Hamlet is cautious and reflective rather than immediately impulsive.
```

## Notes

- The quality of generated answers depends heavily on the selected LLM backend.
- Smaller local models such as TinyLlama are fast to set up but may produce factual errors.
- For stronger answers, use a larger local model through Ollama or connect an OpenAI-compatible API.
- If an old retrieval index fails to load after changing NumPy versions, rebuild it with `python src/build_index.py`.

## Future Improvements

- Add a Streamlit or FastAPI web interface
- Add reranking for retrieved scenes
- Add citation formatting with exact line references
- Improve hallucination resistance with stricter prompt validation
- Add automated retrieval metrics such as recall@k and MRR
- Package the project with a cleaner CLI entry point
```
