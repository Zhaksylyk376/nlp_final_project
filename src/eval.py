from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def summarize(name: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r = {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "qwk": qwk(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    r["confusion"] = confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist()
    return r


def dump_metrics(metrics: dict, filename: str = "metrics.json") -> Path:
    path = RESULTS / filename
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(metrics)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return path


def report_text(y_true, y_pred, labels=(1, 2, 3, 4, 5)) -> str:
    return classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=[str(l) for l in labels], digits=4)
