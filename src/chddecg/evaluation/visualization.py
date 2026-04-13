from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_dashboard(results_path: str | Path, output_path: str | Path) -> str:
    results_path = Path(results_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("CHDdECG Evaluation Dashboard", fontsize=18)

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    metric_names = ["accuracy", "auc", "precision", "recall", "balanced_accuracy", "brier_score"]
    metric_values = [results[name] for name in metric_names]
    ax1.bar(metric_names, metric_values)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_title("Key Metrics")

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    sns.heatmap(np.asarray(results["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title("Confusion Matrix")

    ax3 = plt.subplot2grid((2, 2), (1, 0))
    abnormal = results["classification_report"].get("1", results["classification_report"].get("Abnormal", {}))
    keys = ["precision", "recall", "f1-score"]
    values = [abnormal.get(key, 0.0) for key in keys]
    ax3.bar(keys, values)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title("Positive Class Metrics")

    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax4.axis("off")
    ax4.text(
        0.05,
        0.95,
        "\n".join(
            [
                f"Accuracy: {results['accuracy']:.4f}",
                f"AUC: {results['auc']:.4f}",
                f"Precision: {results['precision']:.4f}",
                f"Recall: {results['recall']:.4f}",
                f"Balanced Accuracy: {results['balanced_accuracy']:.4f}",
                f"Brier Score: {results['brier_score']:.4f}",
            ]
        ),
        va="top",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return str(output_path)
