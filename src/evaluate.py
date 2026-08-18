"""
Evaluation pipeline for the Shakespeare SLM/RAG system.

This module runs both the baseline and RAG systems on the instructor-
provided questions, collects outputs, and produces evaluation results
in CSV format for manual scoring and analysis.

Evaluation criteria (per the assignment specification):
  1. Correctness       — Is the answer factually accurate?
  2. Grounding         — Is the answer supported by retrieved evidence?
  3. Retrieval quality — Are the retrieved passages relevant?
  4. Usefulness        — Would a beginner find this helpful?
  5. Style quality     — Is the language clear and appropriate?

Each criterion is scored 1-5 (1 = poor, 5 = excellent).
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import RESULTS_DIR, DEFAULT_TOP_K
from data_loader import load_instructor_questions
from baseline import baseline_answer
from rag_chatbot import build_rag_system, rag_answer
from chunking import format_chunk_for_display


def run_evaluation(
    output_dir: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    extra_questions: List[Dict[str, str]] | None = None,
) -> Path:
    """
    Run full evaluation pipeline.

    Steps:
      1. Load instructor questions (and optional group-designed questions).
      2. Build the RAG system (retriever + index).
      3. For each question, generate both baseline and RAG answers.
      4. Save results to CSV for manual scoring.
      5. Save raw JSON outputs for programmatic analysis.

    Returns the path to the output directory.
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load questions ────────────────────────────────────────────────
    questions = load_instructor_questions()
    if extra_questions:
        questions.extend(extra_questions)

    print(f"Loaded {len(questions)} evaluation questions.")

    # ── build RAG system ──────────────────────────────────────────────
    print("\nBuilding RAG system ...")
    retriever = build_rag_system(use_saved_index=True)
    print("RAG system ready.\n")

    # ── run evaluation ────────────────────────────────────────────────
    results: List[Dict] = []

    for q in questions:
        qid = q.get("id") or q.get("question_id", "?")
        question = q.get("question", "")
        expected = q.get("expected_focus", "")
        qtype = q.get("type", q.get("question_type", ""))
        play = q.get("play", "")

        print(f"[{qid}] {question}")

        # Baseline
        print("  Running baseline ...", end=" ", flush=True)
        t0 = time.time()
        baseline_resp = baseline_answer(question)
        baseline_time = time.time() - t0
        print(f"({baseline_time:.1f}s)")

        # RAG
        print("  Running RAG ...", end=" ", flush=True)
        t0 = time.time()
        rag_resp, retrieved = rag_answer(question, retriever, top_k=top_k)
        rag_time = time.time() - t0
        print(f"({rag_time:.1f}s)")

        # Format retrieved passages
        passages_text = "\n---\n".join(
            f"[Score: {score:.4f}] "
            + format_chunk_for_display(chunk, include_text=False)
            for chunk, score in retrieved
        )

        # Store results
        for system_name, response, gen_time in [
            ("baseline", baseline_resp, baseline_time),
            ("rag", rag_resp, rag_time),
        ]:
            results.append({
                "question_id": qid,
                "question": question,
                "question_type": qtype,
                "play": play,
                "expected_focus": expected,
                "system": system_name,
                "retrieved_passages": passages_text if system_name == "rag" else "N/A",
                "generated_response": response,
                "generation_time_s": f"{gen_time:.2f}",
                "correctness_score": "",
                "grounding_score": "",
                "retrieval_relevance_score": "",
                "usefulness_score": "",
                "style_quality_score": "",
                "comments": "",
            })

        print()

    # ── save CSV ──────────────────────────────────────────────────────
    csv_path = output_dir / "evaluation_results.csv"
    fieldnames = [
        "question_id", "question", "question_type", "play",
        "expected_focus", "system", "retrieved_passages",
        "generated_response", "generation_time_s",
        "correctness_score", "grounding_score",
        "retrieval_relevance_score", "usefulness_score",
        "style_quality_score", "comments",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"CSV results saved to: {csv_path}")

    # ── save JSON ─────────────────────────────────────────────────────
    json_path = output_dir / "evaluation_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"JSON results saved to: {json_path}")

    # ── print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated: {len(questions)}")
    print(f"Systems compared:    baseline vs. RAG")
    print(f"Output directory:    {output_dir}")
    print()
    print("Next steps:")
    print("  1. Open evaluation_results.csv")
    print("  2. Score each response on the 5 criteria (1-5 scale)")
    print("  3. Add comments noting strengths and failures")
    print("  4. Use the scores for the comparative analysis in your report")

    return output_dir


# ── group-designed questions ──────────────────────────────────────────

GROUP_QUESTIONS = [
    {
        "id": "G1",
        "play": "Macbeth",
        "question": "How does Lady Macbeth manipulate Macbeth into killing Duncan?",
        "expected_focus": "Lady Macbeth questions his manhood, mocks his hesitation, and presents a concrete murder plan.",
        "type": "contextual_qa",
    },
    {
        "id": "G2",
        "play": "Hamlet",
        "question": "What is the significance of the play-within-a-play in Hamlet?",
        "expected_focus": "Hamlet uses The Mousetrap to test Claudius's guilt by re-enacting the murder.",
        "type": "contextual_qa",
    },
    {
        "id": "G3",
        "play": "Romeo and Juliet",
        "question": "Who is Friar Laurence and what role does he play in the tragedy?",
        "expected_focus": "Friar Laurence secretly marries the lovers and devises the sleeping potion plan that goes wrong.",
        "type": "concept_explanation",
    },
    {
        "id": "G4",
        "play": "Macbeth",
        "question": "What do the three witches' prophecies mean and how do they come true?",
        "expected_focus": "The witches predict Macbeth will be Thane of Cawdor and King; Banquo's descendants will be kings. The prophecies drive Macbeth's ambition.",
        "type": "contextual_qa",
    },
    {
        "id": "G5",
        "play": "Romeo and Juliet",
        "question": "Write a short Shakespearean-style soliloquy from Romeo upon first seeing Juliet.",
        "expected_focus": "Creative stylised output capturing Romeo's sudden infatuation, using imagery of light and beauty.",
        "type": "stylised_generation",
    },
]


# ── entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Shakespeare SLM/RAG evaluation")
    parser.add_argument(
        "--include-group", action="store_true",
        help="Include group-designed questions in addition to instructor questions."
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of passages to retrieve (default: {DEFAULT_TOP_K})"
    )
    args = parser.parse_args()

    extra = GROUP_QUESTIONS if args.include_group else None
    run_evaluation(top_k=args.top_k, extra_questions=extra)
