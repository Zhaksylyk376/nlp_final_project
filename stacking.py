"""Validation-only stacking: meta-LR over CatBoost + ruBERT v1 + ruBERT v2 +
TF-IDF LR, plus per-class threshold tuning for weighted F1.

Run after ruBERT v2 completes. Writes submission_stacked.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RNG = 0


def load_or_none(path):
    p = Path(path)
    return np.load(p) if p.exists() else None


def main():
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")

    # Reproduce the same val split used by catboost / rubert
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RNG)
    tri, vai = next(sss.split(tr["text"], tr["rate"]))
    tr_sub = tr.iloc[tri].reset_index(drop=True)
    va_sub = tr.iloc[vai].reset_index(drop=True)
    y_va = va_sub["rate"].values

    print(f"val size: {len(va_sub)}")

    # Load existing model probabilities
    p_cb_v = load_or_none(RES / "catboost_proba_val.npy")
    p_cb_t = load_or_none(RES / "catboost_proba_test.npy")
    p_r1_v = load_or_none(RES / "rubert_proba_val.npy")
    p_r1_t = load_or_none(RES / "rubert_proba_test.npy")
    p_r2_v = load_or_none(RES / "rubert_v2_proba_val.npy")
    p_r2_t = load_or_none(RES / "rubert_v2_proba_test.npy")

    streams_v, streams_t, names = [], [], []
    if p_cb_v is not None:
        streams_v.append(p_cb_v); streams_t.append(p_cb_t); names.append("catboost")
        print(f"  catboost: wF1={f1_score(y_va, p_cb_v.argmax(1)+1, average='weighted'):.4f}")
    if p_r1_v is not None:
        streams_v.append(p_r1_v); streams_t.append(p_r1_t); names.append("rubert_v1")
        print(f"  rubert_v1: wF1={f1_score(y_va, p_r1_v.argmax(1)+1, average='weighted'):.4f}")
    if p_r2_v is not None:
        streams_v.append(p_r2_v); streams_t.append(p_r2_t); names.append("rubert_v2")
        print(f"  rubert_v2: wF1={f1_score(y_va, p_r2_v.argmax(1)+1, average='weighted'):.4f}")

    # Train a quick TF-IDF LR branch and get its val/test probs
    t0 = time.time()
    v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=120_000)
    Xw_tr = v_w.fit_transform(tr_sub["text"]); Xw_va = v_w.transform(va_sub["text"]); Xw_te = v_w.transform(te["text"])
    Xc_tr = v_c.fit_transform(tr_sub["text"]); Xc_va = v_c.transform(va_sub["text"]); Xc_te = v_c.transform(te["text"])
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr(); X_va = hstack([Xw_va, Xc_va]).tocsr(); X_te = hstack([Xw_te, Xc_te]).tocsr()
    lr = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs")
    lr.fit(X_tr, tr_sub["rate"])
    p_lr_v = lr.predict_proba(X_va); p_lr_t = lr.predict_proba(X_te)
    print(f"  LR tfidf: wF1={f1_score(y_va, p_lr_v.argmax(1)+1, average='weighted'):.4f}  ({time.time()-t0:.0f}s)")
    streams_v.append(p_lr_v); streams_t.append(p_lr_t); names.append("tfidf_lr")

    # --- Simple convex blend: grid-search weights on wF1 ---
    print("\n== Convex blend ==")
    best = {"wF1": -1}
    # exhaustive if 2-3 models, random sample if more
    K = len(streams_v)

    def enum_simplex(n, step=0.1):
        if n == 1:
            yield (1.0,)
            return
        for v in np.arange(0.0, 1.0 + 1e-9, step):
            for rest in enum_simplex(n - 1, step):
                if abs(v + sum(rest) - 1.0) < 1e-6 or v + sum(rest) <= 1.0 + 1e-6:
                    pass
                yield (v,) + rest

    # Simpler: random search 10k samples (works for any K)
    rng = np.random.default_rng(0)
    for _ in range(20000):
        w = rng.dirichlet(np.ones(K))
        p = sum(wi * sv for wi, sv in zip(w, streams_v))
        wF1 = f1_score(y_va, p.argmax(1) + 1, average="weighted")
        if wF1 > best["wF1"]:
            best = {"wF1": wF1, "w": w, "proba": p}
    print("Best weights:", {n: round(float(wi), 3) for n, wi in zip(names, best["w"])})
    print(f"Blend val: wF1={best['wF1']:.4f}  acc={accuracy_score(y_va, best['proba'].argmax(1)+1):.4f}  mF1={f1_score(y_va, best['proba'].argmax(1)+1, average='macro'):.4f}")

    # --- Meta-LogReg over stream probabilities ---
    print("\n== Meta-LR ==")
    X_meta_va = np.hstack(streams_v)
    X_meta_te = np.hstack(streams_t)
    # Use validation itself to tune the meta via internal 5-fold OOF; refit on all val for test.
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros((len(y_va), 5))
    for tri_f, vai_f in kf.split(X_meta_va):
        meta = LogisticRegression(C=2.0, max_iter=500)
        meta.fit(X_meta_va[tri_f], y_va[tri_f])
        oof[vai_f] = meta.predict_proba(X_meta_va[vai_f])
    yp_meta = oof.argmax(1) + 1
    wF1_meta = f1_score(y_va, yp_meta, average="weighted")
    print(f"Meta-LR OOF: wF1={wF1_meta:.4f}  acc={accuracy_score(y_va, yp_meta):.4f}")

    # Refit meta on all val, predict test
    meta_full = LogisticRegression(C=2.0, max_iter=500)
    meta_full.fit(X_meta_va, y_va)
    p_meta_t = meta_full.predict_proba(X_meta_te)

    # --- Threshold / class-prior tuning on blend probabilities ---
    print("\n== Per-class bias tuning (weighted F1 optimization) ==")
    # Start from best convex blend; add class-wise log-additive biases.
    base_v = best["proba"]
    base_t = sum(wi * st for wi, st in zip(best["w"], streams_t))

    bias = np.zeros(5)
    best_wF1 = best["wF1"]
    # Simple hill-climbing on 5 scalar biases
    for _ in range(300):
        improved = False
        for cls in range(5):
            for delta in (-0.1, 0.1, -0.03, 0.03):
                trial = bias.copy(); trial[cls] += delta
                p_adj = base_v * np.exp(trial)
                p_adj = p_adj / p_adj.sum(1, keepdims=True)
                wF1_adj = f1_score(y_va, p_adj.argmax(1) + 1, average="weighted")
                if wF1_adj > best_wF1 + 1e-5:
                    bias = trial; best_wF1 = wF1_adj; improved = True
        if not improved:
            break
    print(f"Tuned val: wF1={best_wF1:.4f}  bias={bias.round(3).tolist()}")

    # --- Pick the best stream (convex-blend+bias OR meta-LR) and write submission ---
    candidates = [
        ("blend_bias", best_wF1, base_t * np.exp(bias)),
        ("blend_plain", best["wF1"], base_t),
        ("meta_lr", wF1_meta, p_meta_t),
    ]
    candidates.sort(key=lambda x: -x[1])
    best_name, best_score, best_proba = candidates[0]
    print(f"\nBest head: {best_name}  val wF1={best_score:.4f}")

    pred = best_proba.argmax(1) + 1
    pd.DataFrame({"index": te["index"], "rate": pred.astype(int)}).to_csv(RES / "submission_stacked.csv", index=False)
    print("wrote results/submission_stacked.csv")

    # Also write all three variants for safety
    pd.DataFrame({"index": te["index"], "rate": (base_t * np.exp(bias)).argmax(1).astype(int) + 1}).to_csv(RES / "submission_stacked_biased.csv", index=False)
    pd.DataFrame({"index": te["index"], "rate": p_meta_t.argmax(1).astype(int) + 1}).to_csv(RES / "submission_stacked_metalr.csv", index=False)
    pd.DataFrame({"index": te["index"], "rate": base_t.argmax(1).astype(int) + 1}).to_csv(RES / "submission_stacked_plain.csv", index=False)

    (RES / "stacking_metrics.json").write_text(json.dumps({
        "names": names,
        "blend_weights": [float(x) for x in best["w"]],
        "blend_plain_val_wF1": float(best["wF1"]),
        "blend_biased_val_wF1": float(best_wF1),
        "meta_lr_val_wF1": float(wF1_meta),
        "best_head": best_name,
    }, indent=2))


if __name__ == "__main__":
    main()
