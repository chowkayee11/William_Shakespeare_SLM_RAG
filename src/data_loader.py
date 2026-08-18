"""
Data loading utilities for the Shakespeare SLM/RAG system.

Supports two loading modes:
  1. Scene-level JSONL files  (fast, pre-chunked by the instructor)
  2. Full play JSON files     (more flexible, supports custom chunking)

The loader automatically resolves file locations by checking both
``data/processed/`` and the original dataset directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import PLAY_FILES, SCENE_CHUNK_FILES, DATASET_DIR, PLAY_NAMES, UTTERANCE_FILES


Record = Dict[str, Any]


# ── helpers ───────────────────────────────────────────────────────────

def _extract_scenes(obj: Any) -> List[Record]:
    """Extract scene records from a full play JSON object."""
    if isinstance(obj, dict) and "scenes" in obj:
        return obj["scenes"]
    if isinstance(obj, list):
        return obj
    for key in ["records", "chunks", "data"]:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    raise ValueError("Cannot extract scenes from the loaded JSON object.")


# ── public API ────────────────────────────────────────────────────────

def load_play_json(path: Path) -> Dict[str, Any]:
    """Load a full play JSON file and return the raw dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"Play file not found: {path}\n"
            "Place the provided dataset files in data/processed/ "
            "or ensure the shakespeare_slm_dataset directory exists."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_scene_chunks_jsonl(path: Path) -> List[Record]:
    """Load a scene-level JSONL file (one JSON object per line)."""
    if not path.exists():
        raise FileNotFoundError(f"Scene-chunk file not found: {path}")
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_scene_chunks() -> List[Record]:
    """Load scene-level chunks for all three plays from JSONL files."""
    all_chunks: List[Record] = []
    for name in PLAY_NAMES:
        path = SCENE_CHUNK_FILES[name]
        chunks = load_scene_chunks_jsonl(path)
        for c in chunks:
            c.setdefault("play_key", name)
        all_chunks.extend(chunks)
    return all_chunks


def load_all_plays_scenes() -> List[Record]:
    """Load scene records from all three play JSON files."""
    all_scenes: List[Record] = []
    for name in PLAY_NAMES:
        path = PLAY_FILES[name]
        data = load_play_json(path)
        scenes = _extract_scenes(data)
        for s in scenes:
            s.setdefault("play_key", name)
        all_scenes.extend(scenes)
    return all_scenes


def load_all_utterance_records() -> List[Record]:
    """Load utterance-level records for all three plays from JSONL files."""
    all_utterances: List[Record] = []
    for name in PLAY_NAMES:
        path = UTTERANCE_FILES[name]
        utterances = load_scene_chunks_jsonl(path)
        for u in utterances:
            u.setdefault("play_key", name)
        all_utterances.extend(utterances)
    return all_utterances


def load_instructor_questions() -> List[Dict[str, str]]:
    """Load instructor-provided evaluation questions."""
    from config import INSTRUCTOR_QUESTIONS_PATH, DATASET_QUESTIONS_PATH

    for qpath in [INSTRUCTOR_QUESTIONS_PATH, DATASET_QUESTIONS_PATH]:
        if qpath.exists():
            with qpath.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Handle both list and {"questions": [...]} formats
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
            return data

    raise FileNotFoundError(
        "Instructor questions file not found. "
        "Expected at results/instructor_questions.json or in the dataset dir."
    )


# ── CLI quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Loading scene chunks from JSONL ===")
    chunks = load_all_scene_chunks()
    print(f"Total scene chunks: {len(chunks)}")
    for name in PLAY_NAMES:
        n = sum(1 for c in chunks if c.get("play_key") == name)
        print(f"  {name}: {n} scenes")
    print()

    print("=== Loading scenes from play JSON ===")
    scenes = load_all_plays_scenes()
    print(f"Total scenes: {len(scenes)}")
    print(f"First scene keys: {list(scenes[0].keys())}")
    print()

    print("=== Loading instructor questions ===")
    questions = load_instructor_questions()
    print(f"Total questions: {len(questions)}")
    for q in questions:
        qid = q.get("id") or q.get("question_id", "?")
        print(f"  {qid}: {q.get('question', '')[:60]}...")
