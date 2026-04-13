from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from chddecg.data.datasets import create_tf_dataset, load_processed_arrays
from chddecg.models.resnet_module import InputConv, ResBlock, SE
from chddecg.models.tabnet.custom_objects import AssertFiniteLayer, GroupNormalization, custom_objects, glu, sparsemax
from chddecg.models.tabnet_downsampling import TabNet_downsampling
from chddecg.models.transformer_module import EncoderLayer, MultiHeadAttention, TemporalAttention
from chddecg.training.metrics import F1Score


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "InputConv": InputConv,
            "ResBlock": ResBlock,
            "SE": SE,
            "MultiHeadAttention": MultiHeadAttention,
            "EncoderLayer": EncoderLayer,
            "TemporalAttention": TemporalAttention,
            "glu": glu,
            "sparsemax": sparsemax,
            "GroupNormalization": GroupNormalization,
            "AssertFiniteLayer": AssertFiniteLayer,
            "custom_objects": custom_objects,
            "TabNet_downsampling": TabNet_downsampling,
            "F1Score": F1Score,
        },
        compile=False,
    )


def _save_curve_plot(x, y, xlabel: str, ylabel: str, title: str, save_path: str | Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model(config: dict) -> dict[str, str]:
    processed = load_processed_arrays(config["paths"]["processed_dir"])
    evaluation_dir = Path(config["paths"]["evaluation_dir"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_tf_dataset(
        processed["test_signals"],
        processed["test_clinical"],
        processed["test_features"],
        processed["test_labels"],
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
    )

    model_path = Path(config["paths"]["model_dir"]) / "best_model.keras"
    model = load_trained_model(model_path)
    probs = model.predict(dataset).reshape(-1)

    labels = processed["test_labels"]
    threshold = float(config["evaluation"]["threshold"])
    preds = (probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)

    results = {
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": float(auc(fpr, tpr)),
        "average_precision": float(average_precision_score(labels, probs)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "classification_report": classification_report(labels, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "threshold": threshold,
    }

    with (evaluation_dir / "test_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    _save_curve_plot(fpr, tpr, "False Positive Rate", "True Positive Rate", "ROC Curve", evaluation_dir / "roc_curve.png")
    _save_curve_plot(recall_curve, precision_curve, "Recall", "Precision", "Precision-Recall Curve", evaluation_dir / "pr_curve.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(np.asarray(results["confusion_matrix"]), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(evaluation_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    return {
        "results_json": str(evaluation_dir / "test_results.json"),
        "roc_curve": str(evaluation_dir / "roc_curve.png"),
        "pr_curve": str(evaluation_dir / "pr_curve.png"),
        "confusion_matrix": str(evaluation_dir / "confusion_matrix.png"),
    }
