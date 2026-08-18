"""
RAG chatbot — the main Retrieval-Augmented Generation pipeline.

Architecture
-------------
1. **Load** scene-level chunks from the Shakespeare dataset.
2. **Index** chunks with sentence-transformer embeddings.
3. **Retrieve** top-k relevant passages for a user query.
4. **Prompt** the SLM with the retrieved context + system instruction.
5. **Generate** a grounded answer.

The chatbot supports both interactive (REPL) and programmatic usage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, PROMPT_DIR, CHUNK_STRATEGY
from data_loader import load_all_scene_chunks, load_all_utterance_records
from chunking import create_chunks, format_chunk_for_display, format_chunk_for_prompt
from retrieval import EmbeddingRetriever
from llm_interface import generate


Chunk = Dict[str, Any]


# ── prompt construction ──────────────────────────────────────────────

def load_system_prompt() -> str:
    """Load the system prompt from the prompts directory."""
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    # Fallback default
    return (
        "You are a Shakespeare-aware assistant.\n"
        "Use the retrieved context to answer the user's question.\n"
        "Your answer must be beginner-friendly.\n"
        "If the retrieved context is insufficient, say so clearly.\n"
        "Do not invent unsupported details."
    )


def build_rag_prompt(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    max_context_chars: int = 6000,
) -> str:
    """
    Build a prompt for RAG-based answer generation.

    The prompt includes:
      - retrieved context passages (ranked by relevance)
      - the user's question
      - instructions for grounded answering
    """
    context_blocks: List[str] = []
    total_chars = 0

    for rank, (chunk, score) in enumerate(retrieved, start=1):
        block = format_chunk_for_prompt(chunk)
        if total_chars + len(block) > max_context_chars:
            break
        context_blocks.append(
            f"--- Context {rank} (relevance: {score:.3f}) ---\n{block}"
        )
        total_chars += len(block)

    context = "\n\n".join(context_blocks)

    prompt = (
        f"Retrieved passages from Shakespeare's plays:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Using ONLY the passages above, provide a clear, beginner-friendly answer. "
        f"Cite specific act/scene references when possible. "
        f"If the passages do not contain enough information, say so."
    )
    return prompt


# ── RAG answer generation ────────────────────────────────────────────

def rag_answer(
    query: str,
    retriever: EmbeddingRetriever,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Generate a RAG-grounded answer for a query.

    Returns
    -------
    answer : str
        The generated answer text.
    retrieved : list of (Chunk, float)
        The retrieved passages and their scores.
    """
    retrieved = retriever.retrieve(query, top_k=top_k)
    system_prompt = load_system_prompt()
    user_prompt = build_rag_prompt(query, retrieved)
    answer = generate(user_prompt, system_prompt=system_prompt)
    return answer, retrieved


# ── system initialisation ────────────────────────────────────────────

def build_rag_system(
    mode: str = "dense",
    use_saved_index: bool = True,
) -> EmbeddingRetriever:
    """
    Build (or load) the full RAG retrieval system.

    Parameters
    ----------
    mode : str
        Retrieval mode: 'dense', 'sparse', or 'hybrid'.
    use_saved_index : bool
        If True, try to load a previously saved index before rebuilding.

    Returns
    -------
    EmbeddingRetriever
        A ready-to-use retriever with the index loaded.
    """
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME, mode=mode)

    if use_saved_index:
        try:
            retriever.load_index()
            return retriever
        except FileNotFoundError:
            print("No saved index found. Building from scratch ...")

    if CHUNK_STRATEGY == "scene":
        records = load_all_scene_chunks()
    elif CHUNK_STRATEGY == "utterance_window":
        records = load_all_utterance_records()
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {CHUNK_STRATEGY}")
    chunks = create_chunks(records)
    print(f"Created {len(chunks)} retrieval chunks.")

    retriever.build_index(chunks)

    # Save for next time
    try:
        retriever.save_index()
    except Exception as exc:
        print(f"Warning: could not save index: {exc}")

    return retriever


# ── interactive REPL ──────────────────────────────────────────────────

def interactive_chat() -> None:
    """Run an interactive Q&A loop."""
    print("=" * 70)
    print("  Shakespeare SLM/RAG Chatbot")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 70)
    print()

    retriever = build_rag_system()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer, retrieved = rag_answer(query, retriever)

        print("\n--- Retrieved Evidence ---")
        for rank, (chunk, score) in enumerate(retrieved[:3], 1):
            print(f"\n[Rank {rank} | Score: {score:.4f}]")
            print(format_chunk_for_display(chunk, include_text=False))

        print(f"\n--- Answer ---\n{answer}\n")


# ── entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    interactive_chat()
