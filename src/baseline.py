"""
Baseline system for comparison with the RAG system.

Baseline design
----------------
The baseline sends the user's question directly to the SLM **without
any retrieved context**.  This isolates the model's intrinsic knowledge
of Shakespeare from the grounding effect of RAG retrieval.

By comparing baseline answers to RAG answers we can measure:
  - whether retrieval improves factual correctness;
  - whether grounding reduces hallucination;
  - how much stylistic quality changes with/without context.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from llm_interface import generate


BASELINE_SYSTEM_PROMPT = (
    "You are a helpful assistant with general knowledge of Shakespeare's plays. "
    "Answer the user's question as accurately as you can based on your training data. "
    "If you are unsure, say so. Keep your answer concise (under 200 words)."
)


def baseline_answer(query: str) -> str:
    """
    Generate an answer using only the SLM's parametric knowledge.

    No retrieval context is provided — this is the control condition
    for evaluating whether RAG improves answer quality.
    """
    prompt = (
        f"Question about Shakespeare: {query}\n\n"
        "Please answer based on your knowledge of Shakespeare's plays. "
        "Be specific and cite act/scene numbers if you can."
    )
    return generate(prompt, system_prompt=BASELINE_SYSTEM_PROMPT)


# ── CLI quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "Why does Macbeth kill Duncan?",
        "Who is Hamlet?",
        "What is the conflict between the Montagues and the Capulets?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {baseline_answer(q)}")
        print("-" * 80)
