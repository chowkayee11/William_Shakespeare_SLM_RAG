"""
Build and test the retrieval index.

This is a sanity-check script that verifies the full data pipeline:
  1. Dataset loading
  2. Chunk creation
  3. Embedding generation
  4. Retrieval quality (manual inspection)

Run this first to ensure everything works before using the chatbot
or evaluation pipeline.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, CHUNK_STRATEGY
from data_loader import load_all_scene_chunks, load_all_utterance_records
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


def main() -> None:
    # Step 1: Load data
    print(f"Step 1: Loading data (Strategy: {CHUNK_STRATEGY}) ...")
    if CHUNK_STRATEGY == "scene":
        records = load_all_scene_chunks()
    elif CHUNK_STRATEGY == "utterance_window":
        records = load_all_utterance_records()
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {CHUNK_STRATEGY}")
    print(f"  Loaded {len(records)} records.\n")

    # Step 2: Create chunks
    print("Step 2: Creating retrieval chunks ...")
    chunks = create_chunks(records)
    print(f"  Created {len(chunks)} chunks.\n")

    # Step 3: Build index
    print("Step 3: Building embedding index ...")
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME, mode="dense")
    retriever.build_index(chunks)
    print()

    # Step 4: Save index
    print("Step 4: Saving index to disk ...")
    retriever.save_index()
    print()

    # Step 5: Test retrieval
    test_queries = [
        "Why does Macbeth kill Duncan?",
        "Who is Hamlet?",
        "What is the conflict between the Montagues and the Capulets?",
        "How does Lady Macbeth influence Macbeth?",
        "Why does Hamlet delay his revenge?",
    ]

    print("Step 5: Testing retrieval quality\n")
    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        print(f"Query: {query}")
        for rank, (chunk, score) in enumerate(results, 1):
            play = chunk.get("play", "?")
            act = chunk.get("act", "?")
            scene = chunk.get("scene", "?")
            summary = chunk.get("scene_summary", "")[:80]
            print(f"  #{rank} [{score:.4f}] {play} {act}.{scene} — {summary}")
        print()

    print("=" * 70)
    print("Index build complete. You can now run:")
    print("  python src/rag_chatbot.py    # interactive chatbot")
    print("  python src/evaluate.py       # full evaluation")
    print("=" * 70)


if __name__ == "__main__":
    main()
