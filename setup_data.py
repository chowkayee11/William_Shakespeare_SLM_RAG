"""
Setup script: copy dataset files into the expected data/processed/ directory.

Run this once after cloning the project to set up the data directory.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "shakespeare_slm_dataset" / "shakespeare_slm_dataset"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PLAY_FILES = [
    "hamlet.json",
    "macbeth.json",
    "romeo_and_juliet.json",
    "hamlet_scene_chunks.jsonl",
    "macbeth_scene_chunks.jsonl",
    "romeo_and_juliet_scene_chunks.jsonl",
    "hamlet_utterances.jsonl",
    "macbeth_utterances.jsonl",
    "romeo_and_juliet_utterances.jsonl",
    "instructor_questions.json",
]

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Target directory:  {DATA_DIR}")
    print()

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        print("Please extract the shakespeare_slm_dataset archive first.")
        return

    copied = 0
    for filename in PLAY_FILES:
        src = DATASET_DIR / filename
        if not src.exists():
            print(f"  SKIP  {filename} (not found in dataset)")
            continue

        dst = DATA_DIR / filename
        if dst.exists():
            print(f"  EXISTS {filename}")
        else:
            shutil.copy2(src, dst)
            print(f"  COPIED {filename}")
            copied += 1

    # Also copy instructor questions to results/
    q_src = DATASET_DIR / "instructor_questions.json"
    q_dst = RESULTS_DIR / "instructor_questions.json"
    if q_src.exists() and not q_dst.exists():
        shutil.copy2(q_src, q_dst)
        print(f"  COPIED instructor_questions.json -> results/")
        copied += 1

    print(f"\nDone. {copied} files copied.")


if __name__ == "__main__":
    main()
