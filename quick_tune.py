"""Aggressive bias / weight tuning on existing stacking signals. 2-3 min."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"

tr = pd.read_csv(RAW / "train.csv"); te = pd.read_csv(RAW / "test.csv")
tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
tri, vai = next(sss.split(tr["text"], tr["rate"]))
tr_sub = tr.iloc[tri].reset_index(drop=True); va_sub = tr.iloc[vai].reset_index(drop=True)
y_va = va_sub["rate"].values

p_cb_v = np.load(RES / "catboost_proba_val.npy"); p_cb_t = np.load(RES / "catboost_proba_test.npy")
p_r1_v = np.load(RES / "rubert_proba_val.npy");   p_r1_t = np.load(RES / "rubert_proba_test.npy")

# TF-IDF LR + char
v_w = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True)
v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=120000)
Xw_tr = v_w.fit_transform(tr_sub["text"]); Xw_va = v_w.transform(va_sub["text"]); Xw_te = v_w.transform(te["text"])
Xc_tr = v_c.fit_transform(tr_sub["text"]); Xc_va = v_c.transform(va_sub["text"]); Xc_te = v_c.transform(te["text"])
X_tr = hstack([Xw_tr, Xc_tr]).tocsr(); X_va = hstack([Xw_va, Xc_va]).tocsr(); X_te = hstack([Xw_te, Xc_te]).tocsr()
lr = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs"); lr.fit(X_tr, tr_sub["rate"])
p_lr_v = lr.predict_proba(X_va); p_lr_t = lr.predict_proba(X_te)

# Second LR with balanced
lrb = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs", class_weight="balanced"); lrb.fit(X_tr, tr_sub["rate"])
p_lrb_v = lrb.predict_proba(X_va); p_lrb_t = lrb.predict_proba(X_te)

# Third: pure word LR with different C
lrc = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs"); lrc.fit(Xw_tr, tr_sub["rate"])
p_lrc_v = lrc.predict_proba(Xw_va); p_lrc_t = lrc.predict_proba(Xw_te)

streams_v = [p_cb_v, p_r1_v, p_lr_v, p_lrb_v, p_lrc_v]
streams_t = [p_cb_t, p_r1_t, p_lr_t, p_lrb_t, p_lrc_t]
names = ["catboost", "rubert_v1", "lr_wc", "lr_wc_bal", "lr_w"]
for n, p in zip(names, streams_v):
    print(f"  {n}: wF1={f1_score(y_va, p.argmax(1)+1, average='weighted'):.4f}")

K = len(streams_v)
rng = np.random.default_rng(0)
best = {"wF1": -1}
for _ in range(80000):
    w = rng.dirichlet(np.ones(K))
    p = sum(wi * sv for wi, sv in zip(w, streams_v))
    wF1 = f1_score(y_va, p.argmax(1) + 1, average="weighted")
    if wF1 > best["wF1"]:
        best = {"wF1": wF1, "w": w, "proba_v": p}
print(f"\nBest convex blend: wF1={best['wF1']:.4f}  weights={dict(zip(names, best['w'].round(3).tolist()))}")

# Aggressive bias search: multi-scale hill climbing
bias = np.zeros(5); current = best["wF1"]
for step in (0.3, 0.15, 0.08, 0.04, 0.02, 0.01):
    for _ in range(100):
        improved = False
        for cls in range(5):
            for delta in (-step, step):
                trial = bias.copy(); trial[cls] += delta
                p_adj = best["proba_v"] * np.exp(trial)
                p_adj = p_adj / p_adj.sum(1, keepdims=True)
                s = f1_score(y_va, p_adj.argmax(1)+1, average="weighted")
                if s > current + 1e-5:
                    bias = trial; current = s; improved = True
        if not improved: break
print(f"After bias tuning: wF1={current:.4f}  bias={bias.round(3).tolist()}")

# Apply to test
base_t = sum(wi * st for wi, st in zip(best["w"], streams_t))
p_final = base_t * np.exp(bias)
pred = p_final.argmax(1) + 1
pd.DataFrame({"index": te["index"], "rate": pred.astype(int)}).to_csv(RES / "submission_tuned.csv", index=False)
print("wrote results/submission_tuned.csv")

(RES / "tuned_metrics.json").write_text(json.dumps({"val_wF1_plain": float(best['wF1']), "val_wF1_tuned": float(current), "bias": bias.tolist(), "weights": {n: float(w) for n, w in zip(names, best['w'])}}, indent=2))
