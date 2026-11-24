"""Blend CatBoost and ruBERT probabilities. Grid-search blend weight on
validation weighted-F1, apply to test, write submission.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"

p_cb_v = np.load(RES / "catboost_proba_val.npy")
p_cb_t = np.load(RES / "catboost_proba_test.npy")
p_rb_v = np.load(RES / "rubert_proba_val.npy")
p_rb_t = np.load(RES / "rubert_proba_test.npy")
vai_cb = np.load(RES / "catboost_val_idx.npy")
vai_rb = np.load(RES / "rubert_val_idx.npy")
assert (vai_cb == vai_rb).all(), "val splits don't match!"

tr = pd.read_csv(RAW / "train.csv")
te = pd.read_csv(RAW / "test.csv")
y_val = tr.iloc[vai_cb]["rate"].values

print(f"CatBoost: shape={p_cb_v.shape}  weighted-F1={f1_score(y_val, p_cb_v.argmax(1)+1, average='weighted'):.4f}")
print(f"ruBERT:   shape={p_rb_v.shape}  weighted-F1={f1_score(y_val, p_rb_v.argmax(1)+1, average='weighted'):.4f}")

best = {"wF1": -1}
for w in np.arange(0.0, 1.001, 0.05):
    p = w * p_cb_v + (1 - w) * p_rb_v
    yp = p.argmax(1) + 1
    wF1 = f1_score(y_val, yp, average="weighted")
    if wF1 > best["wF1"]:
        best = {"wF1": wF1, "w": round(float(w), 2),
                "acc": float(accuracy_score(y_val, yp)),
                "mF1": float(f1_score(y_val, yp, average="macro")),
                "qwk": float(cohen_kappa_score(y_val, yp, weights="quadratic"))}

print(f"\nBest blend: CatBoost={best['w']}  ruBERT={round(1-best['w'],2)}")
print(f"  val: acc={best['acc']:.4f}  mF1={best['mF1']:.4f}  wF1={best['wF1']:.4f}  QWK={best['qwk']:.4f}")

p_blend_t = best["w"] * p_cb_t + (1 - best["w"]) * p_rb_t
pred = p_blend_t.argmax(1) + 1
pd.DataFrame({"index": te["index"], "rate": pred.astype(int)}).to_csv(RES / "submission_blend_cb_rubert.csv", index=False)
print("wrote results/submission_blend_cb_rubert.csv")
print(pd.Series(pred).value_counts().sort_index())

(RES / "blend_metrics.json").write_text(json.dumps(best, indent=2))
