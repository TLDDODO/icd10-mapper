"""
evaluator.py
------------
Evaluation metrics for ICD chapter prediction.
Computes: top-1 accuracy, top-3 accuracy, macro F1, confusion matrix.
Compares MiniLM vs ClinicalBERT results side by side.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)
from pathlib import Path


def top_k_accuracy(true_labels: list[str],
                   predictions: list[list[dict]],
                   k: int = 1) -> float:
    """
    Compute top-k accuracy.
    A prediction is correct if the true label appears in the top-k results.

    Args:
        true_labels: ground truth chapter names
        predictions: list of retrieve_top_k outputs (each is list of dicts)
        k: cutoff rank

    Returns:
        accuracy as float (0.0 to 1.0)
    """
    correct = 0
    for true, preds in zip(true_labels, predictions):
        top_k_chapters = [p["chapter"] for p in preds[:k]]
        if true in top_k_chapters:
            correct += 1
    return correct / len(true_labels)


def macro_f1(true_labels: list[str],
             predictions: list[list[dict]]) -> float:
    """
    Macro F1 using top-1 predictions only.
    Macro = unweighted average across all chapters.
    Important when class distribution is imbalanced.
    """
    top1_preds = [preds[0]["chapter"] for preds in predictions]
    return f1_score(true_labels, top1_preds, average="macro", zero_division=0)


def full_classification_report(true_labels: list[str],
                                predictions: list[list[dict]]) -> str:
    """Return sklearn classification report string."""
    top1_preds = [preds[0]["chapter"] for preds in predictions]
    return classification_report(true_labels, top1_preds, zero_division=0)


def plot_confusion_matrix(true_labels: list[str],
                          predictions: list[list[dict]],
                          model_name: str,
                          output_path: str = None):
    """
    Plot and save confusion matrix heatmap.
    """
    top1_preds = [preds[0]["chapter"] for preds in predictions]
    labels = sorted(set(true_labels))

    cm = confusion_matrix(true_labels, top1_preds, labels=labels)

    # Shorten chapter names for display
    short_labels = [l[:25] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Evaluator] Saved confusion matrix → {output_path}")
    plt.close()


def evaluate_model(
    true_labels: list[str],
    predictions: list[list[dict]],
    model_name: str,
    results_dir: str = "results"
) -> dict:
    """
    Run full evaluation for one model. Save results to JSON and plots.

    Returns dict of metrics.
    """
    Path(results_dir).mkdir(exist_ok=True)

    top1 = top_k_accuracy(true_labels, predictions, k=1)
    top3 = top_k_accuracy(true_labels, predictions, k=3)
    f1   = macro_f1(true_labels, predictions)
    report = full_classification_report(true_labels, predictions)

    metrics = {
        "model": model_name,
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "macro_f1": round(f1, 4),
        "n_samples": len(true_labels),
    }

    # Save JSON
    json_path = f"{results_dir}/{model_name.lower().replace(' ', '_')}_eval.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    cm_path = f"{results_dir}/{model_name.lower().replace(' ', '_')}_confusion.png"
    plot_confusion_matrix(true_labels, predictions, model_name, cm_path)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Top-1 Accuracy : {top1:.1%}")
    print(f"  Top-3 Accuracy : {top3:.1%}")
    print(f"  Macro F1       : {f1:.4f}")
    print(f"  Samples        : {len(true_labels)}")
    print(f"\nClassification Report:\n{report}")
    print(f"Results saved → {json_path}")

    return metrics


def compare_models(metrics_list: list[dict]):
    """Print side-by-side comparison of multiple models."""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"{'Model':<20} {'Top-1':>8} {'Top-3':>8} {'Macro F1':>10}")
    print("-"*50)
    for m in metrics_list:
        print(f"{m['model']:<20} {m['top1_accuracy']:>8.1%} "
              f"{m['top3_accuracy']:>8.1%} {m['macro_f1']:>10.4f}")


# ── Quick test with dummy data ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate some predictions
    true = ["Circulatory System", "Respiratory System", "Circulatory System"]
    preds = [
        [{"rank": 1, "chapter": "Circulatory System", "score": 0.9},
         {"rank": 2, "chapter": "Respiratory System", "score": 0.5}],
        [{"rank": 1, "chapter": "Respiratory System", "score": 0.85},
         {"rank": 2, "chapter": "Circulatory System", "score": 0.4}],
        [{"rank": 1, "chapter": "Symptoms, Signs, Ill-defined Conditions", "score": 0.6},
         {"rank": 2, "chapter": "Circulatory System", "score": 0.55}],
    ]

    metrics = evaluate_model(true, preds, model_name="MiniLM_test",
                             results_dir="results")
    print("\nMetrics dict:", metrics)
