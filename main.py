"""
main.py
-------
Full ICD-9 chapter mapping pipeline.

Steps:
  1. Load and preprocess D_ICD_DIAGNOSES.csv
  2. Add chapter labels via ICD-9 code ranges
  3. Train/test split (80/20)
  4. Build chapter embedding index
  5. Run retrieval with MiniLM + ClinicalBERT
  6. Evaluate both models, save results
  7. Optional: launch human review session

Usage:
    python main.py                   # full run, both models
    python main.py --model minilm    # MiniLM only (faster)
    python main.py --review          # human review session only
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")

from preprocess import load_spacy_model, preprocess_dataframe, add_chapter_labels
from embedder import MiniLMEmbedder, ClinicalBERTEmbedder, build_chapter_index
from retriever import batch_retrieve
from evaluator import evaluate_model, compare_models
from human_review import run_review_session


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/raw/D_ICD_DIAGNOSES.csv"
RESULTS_DIR = "results"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K = 3


def load_data() -> pd.DataFrame:
    print(f"[Main] Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"[Main] Loaded {len(df)} ICD entries.")

    # Add chapter labels
    df = add_chapter_labels(df, icd_col="icd9_code")

    # Remove unknowns
    unknown_count = (df["chapter"] == "Unknown").sum()
    if unknown_count > 0:
        print(f"[Main] Removing {unknown_count} entries with unknown chapter.")
    df = df[df["chapter"] != "Unknown"].reset_index(drop=True)

    print(f"[Main] {len(df)} entries with valid chapter labels.")
    print(f"[Main] Chapter distribution:")
    print(df["chapter"].value_counts().to_string())
    return df


def run_pipeline(use_minilm: bool = True,
                 use_clinicalbert: bool = True,
                 run_review: bool = False):

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # ── Step 1: Load and preprocess data ─────────────────────────────────────
    df = load_data()

    print("\n[Main] Running spaCy preprocessing...")
    nlp = load_spacy_model()
    df = preprocess_dataframe(df, text_col="long_title", nlp=nlp)

    # ── Step 2: Train/test split ──────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["chapter"]
    )
    print(f"\n[Main] Split: {len(train_df)} train / {len(test_df)} test")

    # Use norm_text as input to models
    test_texts = test_df["norm_text"].tolist()
    true_labels = test_df["chapter"].tolist()

    all_metrics = []

    # ── Step 3: MiniLM ────────────────────────────────────────────────────────
    if use_minilm:
        print("\n" + "="*50)
        print("MODEL 1: MiniLM (sentence-transformers)")
        print("="*50)
        minilm = MiniLMEmbedder()
        chapter_names, chapter_embs = build_chapter_index(minilm)
        minilm_preds = batch_retrieve(test_texts, minilm, chapter_names, chapter_embs, k=TOP_K)
        metrics = evaluate_model(true_labels, minilm_preds, "MiniLM", RESULTS_DIR)
        all_metrics.append(metrics)

    # ── Step 4: ClinicalBERT ──────────────────────────────────────────────────
    if use_clinicalbert:
        print("\n" + "="*50)
        print("MODEL 2: Bio_ClinicalBERT (transformers)")
        print("="*50)
        clinbert = ClinicalBERTEmbedder()
        chapter_names, chapter_embs = build_chapter_index(clinbert)
        bert_preds = batch_retrieve(test_texts, clinbert, chapter_names, chapter_embs, k=TOP_K)
        metrics = evaluate_model(true_labels, bert_preds, "ClinicalBERT", RESULTS_DIR)
        all_metrics.append(metrics)

    # ── Step 5: Compare ───────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        compare_models(all_metrics)

    # ── Step 6: Human review (optional) ──────────────────────────────────────
    if run_review:
        print("\n[Main] Starting human review session...")
        # Use MiniLM predictions for review if available
        if use_minilm:
            review_preds = minilm_preds
        elif use_clinicalbert:
            review_preds = bert_preds
        else:
            print("[Main] No model run, skipping review.")
            return

        run_review_session(
            test_df.reset_index(drop=True),
            review_preds,
            text_col="long_title",
            label_col="chapter"
        )

    print(f"\n[Main] All done. Results saved to ./{RESULTS_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICD-9 Chapter Mapping Pipeline")
    parser.add_argument(
        "--model",
        choices=["minilm", "clinicalbert", "both"],
        default="both",
        help="Which embedding model to run"
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Launch human review session after evaluation"
    )
    args = parser.parse_args()

    run_pipeline(
        use_minilm=(args.model in ["minilm", "both"]),
        use_clinicalbert=(args.model in ["clinicalbert", "both"]),
        run_review=args.review
    )
