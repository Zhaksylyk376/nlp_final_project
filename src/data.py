from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data"
PROC.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
LABEL2IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    submit: pd.DataFrame


def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    ss = pd.read_csv(RAW / "sample_submission.csv")
    tr["text"] = tr["text"].map(_clean)
    te["text"] = te["text"].map(_clean)
    tr = tr[tr["text"].str.len() >= 2].reset_index(drop=True)
    tr["label"] = tr["rate"].map(LABEL2IDX)
    return tr, te, ss


def stratified_split(df: pd.DataFrame, seed: int = 0, val_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    val_idx = []
    for lab, g in df.groupby("label"):
        n = max(1, int(len(g) * val_frac))
        val_idx.extend(rng.choice(g.index.values, size=n, replace=False).tolist())
    val_mask = df.index.isin(val_idx)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)


def load_split(seed: int = 0, val_frac: float = 0.1) -> Split:
    tr_full, te, ss = load_raw()
    tr, va = stratified_split(tr_full, seed=seed, val_frac=val_frac)
    return Split(train=tr, val=va, test=te, submit=ss)


def kfold_indices(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df["text"].values, df["label"].values))


def write_submission(index: pd.Series | np.ndarray, pred_class_1_to_5: np.ndarray, filename: str = "submission.csv") -> Path:
    out = PROC.parent / "results" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.asarray(index), "rate": np.asarray(pred_class_1_to_5, dtype=int)}).to_csv(out, index=False)
    return out
