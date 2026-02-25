"""
human_review.py
---------------
CLI human-in-the-loop validation tool.
Shows model top-3 predictions, asks human to select correct chapter.
Saves review log to CSV for audit trail.

Usage:
    python src/human_review.py
"""

import csv
import os
import pandas as pd
from datetime import datetime
from pathlib import Path


REVIEW_LOG_PATH = "results/human_review_log.csv"
LOG_COLUMNS = ["timestamp", "input_text", "true_chapter",
               "pred_1", "score_1", "pred_2", "score_2", "pred_3", "score_3",
               "human_choice", "human_chapter", "model_correct", "human_correct"]


def init_review_log():
    """Create log file with headers if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)
    if not os.path.exists(REVIEW_LOG_PATH):
        with open(REVIEW_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
        print(f"[Review] Created log file → {REVIEW_LOG_PATH}")


def append_review_log(row: dict):
    """Append one review result to the log CSV."""
    with open(REVIEW_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)


def display_prediction(text: str, true_chapter: str,
                        predictions: list[dict]):
    """Print formatted prediction display."""
    print("\n" + "─"*60)
    print(f"INPUT TEXT:\n  {text}")
    print(f"\nTRUE CHAPTER:  {true_chapter}")
    print("\nMODEL PREDICTIONS:")
    for p in predictions:
        marker = "✓" if p["chapter"] == true_chapter else " "
        print(f"  [{p['rank']}] {marker} {p['chapter']:<45}  score={p['score']:.3f}")
    print()


def get_human_input(predictions: list[dict]) -> tuple[str, str]:
    """
    Ask human to choose the correct chapter.
    Returns (choice_str, selected_chapter_name).
    """
    while True:
        choice = input("Your choice [1/2/3/s=skip/q=quit]: ").strip().lower()

        if choice == "q":
            return "quit", ""

        if choice == "s":
            return "skip", ""

        if choice in ["1", "2", "3"]:
            idx = int(choice) - 1
            if idx < len(predictions):
                return choice, predictions[idx]["chapter"]

        print("  → Please enter 1, 2, 3, s (skip), or q (quit).")


def run_review_session(
    test_df: pd.DataFrame,
    predictions: list[list[dict]],
    text_col: str = "long_title",
    label_col: str = "chapter",
    max_samples: int = 50
):
    """
    Run interactive review session.

    Args:
        test_df: dataframe with text and true labels
        predictions: model predictions from batch_retrieve
        text_col: column name for input text
        label_col: column name for true chapter
        max_samples: max number of samples to review
    """
    init_review_log()

    texts = test_df[text_col].tolist()
    true_labels = test_df[label_col].tolist()

    n = min(len(texts), max_samples)
    reviewed = 0
    agreed = 0

    print("\n" + "="*60)
    print("HUMAN-IN-THE-LOOP REVIEW SESSION")
    print(f"Reviewing up to {n} samples. Type q to quit at any time.")
    print("="*60)

    for i in range(n):
        text = texts[i]
        true = true_labels[i]
        preds = predictions[i]

        display_prediction(text, true, preds)

        choice, human_chapter = get_human_input(preds)

        if choice == "quit":
            print("\n[Review] Session ended by user.")
            break

        if choice == "skip":
            print("  → Skipped.")
            continue

        # Log the result
        model_correct = (preds[0]["chapter"] == true)
        human_correct = (human_chapter == true)

        if human_correct:
            agreed += 1
        reviewed += 1

        row = {
            "timestamp": datetime.now().isoformat(),
            "input_text": text[:200],
            "true_chapter": true,
            "pred_1": preds[0]["chapter"] if len(preds) > 0 else "",
            "score_1": preds[0]["score"] if len(preds) > 0 else 0,
            "pred_2": preds[1]["chapter"] if len(preds) > 1 else "",
            "score_2": preds[1]["score"] if len(preds) > 1 else 0,
            "pred_3": preds[2]["chapter"] if len(preds) > 2 else "",
            "score_3": preds[2]["score"] if len(preds) > 2 else 0,
            "human_choice": choice,
            "human_chapter": human_chapter,
            "model_correct": model_correct,
            "human_correct": human_correct,
        }
        append_review_log(row)

        status = "✓ Correct" if human_correct else "✗ Incorrect"
        print(f"  → Logged: {status}")

    # Session summary
    if reviewed > 0:
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"  Reviewed : {reviewed}")
        print(f"  Correct  : {agreed} ({agreed/reviewed:.1%})")
        print(f"  Log saved: {REVIEW_LOG_PATH}")


def review_summary():
    """Print summary statistics from existing review log."""
    if not os.path.exists(REVIEW_LOG_PATH):
        print("No review log found.")
        return

    df = pd.read_csv(REVIEW_LOG_PATH)
    print(f"\n=== Review Log Summary ===")
    print(f"Total reviewed  : {len(df)}")
    print(f"Human accuracy  : {df['human_correct'].mean():.1%}")
    print(f"Model accuracy  : {df['model_correct'].mean():.1%}")
    print(f"\nDisagreements (human overrode model):")
    disagreements = df[df["human_correct"] & ~df["model_correct"]]
    if len(disagreements):
        for _, row in disagreements.iterrows():
            print(f"  Text   : {row['input_text'][:60]}...")
            print(f"  Model  : {row['pred_1']}")
            print(f"  Human  : {row['human_chapter']}")
            print()
    else:
        print("  None found.")


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    from preprocess import load_spacy_model, preprocess_dataframe, add_chapter_labels
    from embedder import MiniLMEmbedder, build_chapter_index
    from retriever import batch_retrieve

    import pandas as pd

    print("Loading data...")
    icd_df = pd.read_csv("data/raw/D_ICD_DIAGNOSES.csv")
    icd_df = add_chapter_labels(icd_df)
    icd_df = icd_df[icd_df["chapter"] != "Unknown"].reset_index(drop=True)

    # Use a small random sample for review
    sample = icd_df.sample(n=20, random_state=99).reset_index(drop=True)

    print("Building embeddings...")
    embedder = MiniLMEmbedder()
    chapter_names, chapter_embeddings = build_chapter_index(embedder)
    preds = batch_retrieve(
        sample["long_title"].tolist(),
        embedder, chapter_names, chapter_embeddings
    )

    run_review_session(sample, preds, text_col="long_title", label_col="chapter")
    review_summary()
