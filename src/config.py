"""
Configuration for the Shakespeare SLM/RAG System.

This module centralises all paths, model names, and hyper-parameters
so that every other module can import a single source of truth.

Design decisions:
  - Scene-level chunks are used as the default retrieval unit because
    each scene forms a coherent dramatic episode and keeps chunk sizes
    manageable for both the embedding model and the SLM context window.
  - all-MiniLM-L6-v2 is chosen as the embedding model for its balance
    between embedding quality and low resource footprint.
  - TinyLlama-1.1B-Chat is chosen as the default SLM because it fits
    comfortably in <=4 GB VRAM and supports chat-template prompts.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_DIR = PROJECT_ROOT / "shakespeare_slm_dataset" / "shakespeare_slm_dataset"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Play files -- we look in data/processed first, then fall back to the
# dataset directory so that users do not have to copy files manually.
PLAY_NAMES = ["hamlet", "macbeth", "romeo_and_juliet"]

def _resolve_play_path(name: str) -> Path:
    """Return the first existing path for a play JSON file."""
    for base in [DATA_DIR, DATASET_DIR]:
        p = base / f"{name}.json"
        if p.exists():
            return p
    # If neither exists yet, return the expected path under data/processed
    return DATA_DIR / f"{name}.json"

PLAY_FILES = {name: _resolve_play_path(name) for name in PLAY_NAMES}

# Scene-chunk JSONL files (pre-built by the instructor)
def _resolve_scene_chunks_path(name: str) -> Path:
    for base in [DATA_DIR, DATASET_DIR]:
        p = base / f"{name}_scene_chunks.jsonl"
        if p.exists():
            return p
    return DATASET_DIR / f"{name}_scene_chunks.jsonl"

SCENE_CHUNK_FILES = {name: _resolve_scene_chunks_path(name) for name in PLAY_NAMES}

# Utterance JSONL files
def _resolve_utterances_path(name: str) -> Path:
    for base in [DATA_DIR, DATASET_DIR]:
        p = base / f"{name}_utterances.jsonl"
        if p.exists():
            return p
    return DATA_DIR / f"{name}_utterances.jsonl"

UTTERANCE_FILES = {name: _resolve_utterances_path(name) for name in PLAY_NAMES}

# Instructor questions
INSTRUCTOR_QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
DATASET_QUESTIONS_PATH = DATASET_DIR / "instructor_questions.json"

# ---------------------------------------------------------------------------
# Retrieval hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5

# Embedding model (sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# SLM / Generation settings
# ---------------------------------------------------------------------------
# Supported backends: "huggingface", "ollama", "openai_compatible"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "huggingface")

# HuggingFace local model
HF_MODEL_NAME = os.environ.get(
    "HF_MODEL_NAME",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# Ollama settings (if LLM_BACKEND == "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")

# OpenAI-compatible API (if LLM_BACKEND == "openai_compatible")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "local-model")

# Generation hyper-parameters
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
# Supported strategies: "scene", "utterance_window"
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "scene")

# Utterance windowing parameters (used if CHUNK_STRATEGY == "utterance_window")
UTTERANCE_WINDOW_SIZE = int(os.environ.get("UTTERANCE_WINDOW_SIZE", 8))
UTTERANCE_WINDOW_OVERLAP = int(os.environ.get("UTTERANCE_WINDOW_OVERLAP", 2))

# Whether to use pre-built scene JSONL files (True) or build chunks from
# the full play JSON (False). Only applies if CHUNK_STRATEGY == "scene".
USE_SCENE_JSONL = True

# Whether to enrich chunk text with metadata (play, act, scene, summary)
# before embedding.  This improves retrieval for metadata-aware queries.
ENRICH_CHUNK_TEXT = True
