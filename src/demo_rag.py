"""
Demo script to test the Shakespeare SLM/RAG chatbot using the speaker-turn sliding window chunking strategy.
Runs a few typical queries to demonstrate the retrieval and generation effects.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_chatbot import build_rag_system, rag_answer
from chunking import format_chunk_for_display

DEMO_QUESTIONS = [
    "How does Lady Macbeth manipulate Macbeth into killing Duncan?",
    "What is the significance of the play-within-a-play in Hamlet?",
    "Who is Friar Laurence and what role does he play in the tragedy?"
]

def run_demo():
    print("=" * 80)
    print("  Starting Shakespeare RAG Demo (Speaker-Turn Sliding Window Chunking)")
    print("=" * 80)
    print()

    # Build or load RAG retriever
    print("Building/loading retriever...")
    retriever = build_rag_system(use_saved_index=True)
    print("Retriever ready.\n")

    for idx, query in enumerate(DEMO_QUESTIONS, 1):
        print("-" * 80)
        print(f"Demo Query {idx}: {query}")
        print("-" * 80)
        
        print("Retrieving and generating answer...")
        answer, retrieved = rag_answer(query, retriever, top_k=3)
        
        print("\n[Retrieved Evidence]")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print(f"\nRank {rank} (score: {score:.4f}):")
            print(format_chunk_for_display(chunk, include_text=True))
            print("~" * 40)
            
        print("\n[Generated Response]")
        print(answer)
        print()

if __name__ == "__main__":
    run_demo()
