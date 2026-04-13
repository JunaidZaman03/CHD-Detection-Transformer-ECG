from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from chddecg.data.datasets import load_processed_arrays
from chddecg.evaluation.evaluate import load_trained_model


def run_lead_ablation_test(config: dict) -> dict[str, str]:
    processed = load_processed_arrays(config["paths"]["processed_dir"])
    model_path = Path(config["paths"]["model_dir"]) / "best_model.keras"
    output_dir = Path(config["paths"]["perturbation_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(model_path)
    signals = processed["test_signals"].astype(np.float16)
    clinical = processed["test_clinical"].astype(np.float16)
    handcrafted = processed["test_features"].astype(np.float16)
    labels = processed["test_labels"]

    baseline_probs = model.predict(
        {
            "signal_input": signals,
            "clinical_input": clinical,
            "wavelet_input": handcrafted,
        },
        verbose=0,
    ).reshape(-1)

    baseline = {
        "auc": float(roc_auc_score(labels, baseline_probs)),
        "accuracy": float(accuracy_score(labels, (baseline_probs >= 0.5).astype(int))),
    }

    lead_results: list[dict] = []
    for lead_idx in range(signals.shape[-1]):
        perturbed = signals.copy()
        perturbed[:, :, lead_idx] = 0.0

        probs = model.predict(
            {
                "signal_input": perturbed,
                "clinical_input": clinical,
                "wavelet_input": handcrafted,
            },
            verbose=0,
        ).reshape(-1)

        auc_value = float(roc_auc_score(labels, probs))
        acc_value = float(accuracy_score(labels, (probs >= 0.5).astype(int)))
        lead_results.append(
            {
                "lead_index": int(lead_idx),
                "auc": auc_value,
                "accuracy": acc_value,
                "delta_auc": auc_value - baseline["auc"],
                "delta_accuracy": acc_value - baseline["accuracy"],
            }
        )

    payload = {"baseline": baseline, "lead_ablation": lead_results}
    output_path = output_dir / "lead_ablation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {"perturbation_json": str(output_path)}
