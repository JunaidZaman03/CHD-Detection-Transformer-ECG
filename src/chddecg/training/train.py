from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import AdamW

from chddecg.data.datasets import create_tf_dataset, load_processed_arrays
from chddecg.models import CHDdECG
from chddecg.training.metrics import F1Score


def _plot_history(history: tf.keras.callbacks.History, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for key, values in history.history.items():
        if key.startswith("val_"):
            continue
        plt.plot(values, label=key)
        val_key = f"val_{key}"
        if val_key in history.history:
            plt.plot(history.history[val_key], label=val_key)
    plt.xlabel("Epoch")
    plt.ylabel("Metric / Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_model(config: dict) -> dict[str, str]:
    processed = load_processed_arrays(config["paths"]["processed_dir"])
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(config["training"]["batch_size"])
    train_ds = create_tf_dataset(
        processed["train_signals"],
        processed["train_clinical"],
        processed["train_features"],
        processed["train_labels"],
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = create_tf_dataset(
        processed["val_signals"],
        processed["val_clinical"],
        processed["val_features"],
        processed["val_labels"],
        batch_size=batch_size,
        shuffle=False,
    )

    model = CHDdECG(
        num_classes=2,
        use_tabnet=bool(config["training"]["use_tabnet"]),
        use_attention=bool(config["training"]["use_attention"]),
    )

    loss = BinaryCrossentropy(label_smoothing=float(config["training"]["label_smoothing"]))
    optimizer = AdamW(learning_rate=float(config["training"]["learning_rate"]))

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            F1Score(name="f1_score"),
        ],
    )

    class_weight = None
    if bool(config["training"]["use_class_weights"]):
        labels = processed["train_labels"]
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    checkpoint_path = model_dir / "best_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=int(config["training"]["early_stopping_patience"]), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_auc", mode="max", save_best_only=True),
        CSVLogger(model_dir / "training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["training"]["epochs"]),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    _plot_history(history, model_dir / "training_history.png")

    with (model_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)

    val_probs = model.predict(val_ds).reshape(-1)
    val_preds = (val_probs >= 0.5).astype(int)
    val_labels = processed["val_labels"]

    report = {
        "classification_report": classification_report(val_labels, val_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(val_labels, val_preds).tolist(),
    }
    with (model_dir / "val_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return {
        "model_path": str(checkpoint_path),
        "history_path": str(model_dir / "history.json"),
        "plot_path": str(model_dir / "training_history.png"),
        "report_path": str(model_dir / "val_report.json"),
    }
