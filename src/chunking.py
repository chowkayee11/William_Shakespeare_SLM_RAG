"""
Chunking utilities for the Shakespeare SLM/RAG system.

Chunking strategy justification
--------------------------------
We use **scene-level chunks** as the primary retrieval unit.

Rationale:
  1. Each scene is a self-contained dramatic episode with a clear narrative
     purpose, making it a natural unit for question answering.
  2. Scene-level chunks keep the average token count within the context
     window of lightweight SLMs (most scenes are 200–800 tokens).
  3. The instructor-provided scene JSONL files already include curated
     ``scene_summary`` and ``keywords`` fields that we prepend to the
     chunk text during enrichment, significantly improving retrieval
     accuracy for thematic and character-level queries.
  4. Compared to utterance-level chunks, scene-level chunks preserve
     conversational flow and stage directions, which are often essential
     for answering "why" questions.

We also support **enriched chunks** that prepend metadata (play name,
act/scene numbers, location, summary, keywords) to the raw scene text.
This allows the embedding model to capture metadata signals without
requiring a separate metadata filter.
"""

from __future__ import annotations

from typing import Any, Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import ENRICH_CHUNK_TEXT, CHUNK_STRATEGY, UTTERANCE_WINDOW_SIZE, UTTERANCE_WINDOW_OVERLAP


Record = Dict[str, Any]
Chunk = Dict[str, Any]


def _build_enriched_text(record: Record) -> str:
    """
    Build a text representation that combines metadata and scene text.

    The enriched format places high-level metadata first so that the
    embedding model can capture thematic signals even for short queries.
    """
    parts: List[str] = []

    play = record.get("play", record.get("play_key", ""))
    act = record.get("act", "")
    scene = record.get("scene", "")
    location = record.get("location", "")
    summary = record.get("scene_summary", "")
    keywords = record.get("keywords", [])

    header = f"Play: {play}"
    if act:
        header += f", Act {act}"
    if scene:
        header += f", Scene {scene}"
    if location:
        header += f" — {location}"
    parts.append(header)

    if summary:
        parts.append(f"Summary: {summary}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    text = record.get("text", "")
    if text:
        parts.append(text)

    return "\n".join(parts)


def _get_raw_text(record: Record) -> str:
    """Extract raw text from a record."""
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def create_chunks(records: List[Record]) -> List[Chunk]:
    """
    Convert structured scene records or utterance records into retrieval chunks.

    Each chunk contains:
      - chunk_id:  unique identifier
      - play:      play name
      - act/scene: act and scene numbers
      - speaker:   speaker name(s)
      - text:      the text used for embedding (enriched or raw)
      - raw_text:  the original text snippet
      - metadata:  full context metadata dictionary
    """
    chunks: List[Chunk] = []

    if CHUNK_STRATEGY == "scene":
        for i, record in enumerate(records):
            raw_text = _get_raw_text(record)
            if not raw_text:
                continue

            if ENRICH_CHUNK_TEXT:
                embed_text = _build_enriched_text(record)
            else:
                embed_text = raw_text

            chunk: Chunk = {
                "chunk_id": (
                    record.get("scene_id")
                    or record.get("source_id")
                    or record.get("id")
                    or f"chunk_{i:06d}"
                ),
                "play": record.get("play", record.get("play_key", "unknown")),
                "act": record.get("act"),
                "scene": record.get("scene"),
                "location": record.get("location", ""),
                "scene_summary": record.get("scene_summary", ""),
                "keywords": record.get("keywords", []),
                "speaker": record.get("speaker"),
                "text": embed_text,       # used for embedding & retrieval
                "raw_text": raw_text,      # original text for display
                "metadata": record,
            }
            chunks.append(chunk)

    elif CHUNK_STRATEGY == "utterance_window":
        from collections import defaultdict
        # Group utterances by (play, act, scene)
        scenes_map = defaultdict(list)
        for record in records:
            play = record.get("play", record.get("play_key", "unknown"))
            act = record.get("act", "?")
            scene = record.get("scene", "?")
            scenes_map[(play, act, scene)].append(record)

        for (play, act, scene), scene_records in scenes_map.items():
            n_records = len(scene_records)
            if n_records == 0:
                continue

            # Step size for sliding window
            step = max(1, UTTERANCE_WINDOW_SIZE - UTTERANCE_WINDOW_OVERLAP)

            idx = 0
            while idx < n_records:
                end_idx = min(idx + UTTERANCE_WINDOW_SIZE, n_records)
                window = scene_records[idx:end_idx]

                # Format dialogue text within the window
                formatted_lines = []
                for u in window:
                    sp = u.get("speaker", "")
                    ut_text = u.get("text", "").strip()
                    if not ut_text:
                        continue
                    if sp in ("STAGE_DIRECTION", "STAGE"):
                        formatted_lines.append(f"[{ut_text}]")
                    elif sp in ("ELSINORE", "FLOURISH", "FAREWELL"):
                        formatted_lines.append(ut_text)
                    else:
                        formatted_lines.append(f"{sp}: {ut_text}")

                raw_text = "\n".join(formatted_lines)
                if not raw_text.strip():
                    idx += step
                    continue

                # Retrieve common metadata from the first record in the window
                first_rec = window[0]
                location = first_rec.get("location", "")
                summary = first_rec.get("scene_summary", "")
                keywords = first_rec.get("keywords", [])

                # Build enriched text (metadata + dialogue) if enabled
                if ENRICH_CHUNK_TEXT:
                    parts = []
                    header = f"Play: {play}"
                    if act:
                        header += f", Act {act}"
                    if scene:
                        header += f", Scene {scene}"
                    if location:
                        header += f" — {location}"
                    parts.append(header)
                    if summary:
                        parts.append(f"Summary: {summary}")
                    if keywords:
                        parts.append(f"Keywords: {', '.join(keywords)}")
                    parts.append(raw_text)
                    embed_text = "\n".join(parts)
                else:
                    embed_text = raw_text

                chunk_id = f"{play.lower().replace(' ', '_')}_act_{act}_scene_{scene}_u{idx}_to{end_idx-1}"

                chunk: Chunk = {
                    "chunk_id": chunk_id,
                    "play": play,
                    "act": act,
                    "scene": scene,
                    "location": location,
                    "scene_summary": summary,
                    "keywords": keywords,
                    "speaker": list(set([u.get("speaker") for u in window if u.get("speaker") not in ("STAGE_DIRECTION", "STAGE")])),
                    "text": embed_text,
                    "raw_text": raw_text,
                    "metadata": {
                        "play": play,
                        "act": act,
                        "scene": scene,
                        "location": location,
                        "scene_summary": summary,
                        "keywords": keywords,
                        "utterance_start_idx": idx,
                        "utterance_end_idx": end_idx - 1,
                        "window_utterances": window
                    }
                }
                chunks.append(chunk)

                if end_idx == n_records:
                    break
                idx += step

    return chunks


def format_chunk_for_display(chunk: Chunk, include_text: bool = True) -> str:
    """Format a retrieved chunk for human-readable display."""
    play = chunk.get("play", "Unknown")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    location = chunk.get("location", "")
    summary = chunk.get("scene_summary", "")

    header = f"{play}, Act {act}, Scene {scene}"
    if location:
        header += f" — {location}"

    lines = [f"[{header}]"]
    if summary:
        lines.append(f"  Summary: {summary}")

    if include_text:
        raw = chunk.get("raw_text", chunk.get("text", ""))
        # Truncate very long scenes for display
        if len(raw) > 1500:
            raw = raw[:1500] + "\n  [...truncated...]"
        lines.append(raw)

    return "\n".join(lines)


def format_chunk_for_prompt(chunk: Chunk, max_chars: int = 2000) -> str:
    """
    Format a chunk for inclusion in an LLM prompt.

    Shorter than display format to conserve context window tokens.
    """
    play = chunk.get("play", "Unknown")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")
    summary = chunk.get("scene_summary", "")

    header = f"[{play}, Act {act}, Scene {scene}]"
    if summary:
        header += f" {summary}"

    raw = chunk.get("raw_text", chunk.get("text", ""))
    if len(raw) > max_chars:
        raw = raw[:max_chars] + " [...truncated...]"

    return f"{header}\n{raw}"


# ── CLI quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_all_scene_chunks

    records = load_all_scene_chunks()
    chunks = create_chunks(records)
    print(f"Created {len(chunks)} chunks from {len(records)} records.")
    print()
    print("=== Sample chunk (enriched) ===")
    print(chunks[0]["text"][:500])
    print()
    print("=== Formatted for display ===")
    print(format_chunk_for_display(chunks[2]))
