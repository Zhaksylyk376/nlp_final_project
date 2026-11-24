"""Produce the strongest CPU-only submission we can without transformers.
Validates on 10% stratified hold-out, prints metrics, then retrains all models
on 100% of train and writes the final submission.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

RNG = 0
NUM_CLASSES = 5


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def eround(proba, classes):
    """Expected-rating decoding; Bayes-optimal under squared ordinal loss."""
    E = proba @ classes
    return np.clip(np.rint(E).astype(int), classes.min(), classes.max())


_WORD = re.compile(r"[А-Яа-яЁё]+")
_SENT = re.compile(r"[.!?…]+")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
RU_FUNC = set("и в во не на с со а по от у но из к о до за же ли или ни то что чтобы если же вот тут там так как кто я ты он она оно мы вы они меня тебя его её нас вас их мне тебе ему ей нам вам им".split())
POS_CUE = ("хорош", "отличн", "супер", "вкусн", "рекоменд", "удобн", "любл", "спасиб", "дружелюбн", "вежлив", "прият", "улыбч", "быстр", "свеж")
NEG_CUE = ("плох", "ужас", "отврат", "грязн", "хам", "обман", "дорог", "никогда", "ненавиж", "медленн", "груб", "очеред", "просроч", "испорч")


def stylo(texts):
    rows = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        toks = [w.lower() for w in _WORD.findall(t)]
        n = max(len(toks), 1)
        sents = [s for s in _SENT.split(t) if s.strip()] or [t]
        sl = [len(_WORD.findall(s)) for s in sents] or [n]
        func = sum(1 for w in toks if w in RU_FUNC)
        up = sum(1 for c in t if c.isupper())
        excl = t.count("!"); quest = t.count("?")
        emoji = len(_EMOJI.findall(t))
        pos = sum(t.lower().count(w) for w in POS_CUE)
        neg = sum(t.lower().count(w) for w in NEG_CUE)
        rows.append([
            len(t), n, len(set(toks)) / n,
            float(np.mean(sl)), float(np.std(sl)),
            func / n, up / max(len(t), 1),
            excl / n, quest / n, emoji,
            pos, neg, pos - neg,
        ])
    cols = ["n_chars", "n_tokens", "ttr", "avg_sl", "std_sl", "func_r", "up_r", "excl_r", "quest_r", "emoji", "pos_cue", "neg_cue", "net_cue"]
    return pd.DataFrame(rows, columns=cols)


def report(name, y, yp):
    r = {
        "name": name,
        "acc": round(accuracy_score(y, yp), 4),
        "macro_f1": round(f1_score(y, yp, average="macro"), 4),
        "qwk": round(qwk(y, yp), 4),
        "mae": round(mean_absolute_error(y, yp), 4),
    }
    print(f"  {name:30s} acc={r['acc']:.4f}  mF1={r['macro_f1']:.4f}  QWK={r['qwk']:.4f}  MAE={r['mae']:.3f}")
    return r


def main():
    t0 = time.time()
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")
    print(f"load: {time.time()-t0:.1f}s  train={len(tr)}  test={len(te)}")

    # ---- Stratified hold-out for validation ----
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RNG)
    train_idx, val_idx = next(sss.split(tr["text"], tr["rate"]))
    tr_sub = tr.iloc[train_idx].reset_index(drop=True)
    va_sub = tr.iloc[val_idx].reset_index(drop=True)
    print(f"  split: train_sub={len(tr_sub)}  val={len(va_sub)}")

    # ---- TF-IDF features ----
    t0 = time.time()
    v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=200_000)

    Xw_tr = v_w.fit_transform(tr_sub["text"]); Xw_va = v_w.transform(va_sub["text"])
    Xc_tr = v_c.fit_transform(tr_sub["text"]); Xc_va = v_c.transform(va_sub["text"])
    X_tfidf_tr = hstack([Xw_tr, Xc_tr]).tocsr(); X_tfidf_va = hstack([Xw_va, Xc_va]).tocsr()
    print(f"tfidf (val fit): {time.time()-t0:.1f}s  shape={X_tfidf_tr.shape}")

    # ---- Model 1: Logistic regression on TF-IDF ----
    t0 = time.time()
    lr = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs", class_weight="balanced")
    lr.fit(X_tfidf_tr, tr_sub["rate"])
    proba_lr_va = lr.predict_proba(X_tfidf_va)
    print(f"LR fit: {time.time()-t0:.1f}s")

    # ---- SVD + stylometry ----
    t0 = time.time()
    svd = TruncatedSVD(n_components=200, random_state=RNG)
    Xsvd_tr = svd.fit_transform(X_tfidf_tr); Xsvd_va = svd.transform(X_tfidf_va)
    S_tr = stylo(tr_sub["text"]).values; S_va = stylo(va_sub["text"]).values
    X_dense_tr = np.hstack([Xsvd_tr, S_tr]); X_dense_va = np.hstack([Xsvd_va, S_va])
    print(f"SVD+stylo (val fit): {time.time()-t0:.1f}s  shape={X_dense_tr.shape}")

    # ---- Model 2: LightGBM ----
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, class_weight="balanced", random_state=RNG, verbose=-1)
    lgbm.fit(X_dense_tr, tr_sub["rate"])
    proba_lgb_va = lgbm.predict_proba(X_dense_va)
    print(f"LGBM fit: {time.time()-t0:.1f}s")

    # ---- Model 3: XGBoost ----
    t0 = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
        colsample_bytree=0.8, random_state=RNG, n_jobs=-1, eval_metric="mlogloss",
    )
    # XGBoost labels must be 0..K-1
    xgbm.fit(X_dense_tr, tr_sub["rate"].values - 1)
    proba_xgb_va = xgbm.predict_proba(X_dense_va)
    print(f"XGB fit: {time.time()-t0:.1f}s")

    classes = lr.classes_  # 1..5

    # ---- Validation metrics ----
    print("\n== Validation (10% hold-out) ==")
    y_va = va_sub["rate"].values
    r1a = report("LR    (argmax)", y_va, classes[proba_lr_va.argmax(1)])
    r1e = report("LR    (E-round)", y_va, eround(proba_lr_va, classes))
    r2a = report("LGBM  (argmax)", y_va, classes[proba_lgb_va.argmax(1)])
    r2e = report("LGBM  (E-round)", y_va, eround(proba_lgb_va, classes))
    r3a = report("XGB   (argmax)", y_va, classes[proba_xgb_va.argmax(1)])
    r3e = report("XGB   (E-round)", y_va, eround(proba_xgb_va, classes))

    # Convex blend search (grid) using accuracy as reference, QWK as secondary.
    best = {"metric": -1}
    for w_lr in np.arange(0.0, 1.01, 0.1):
        for w_lg in np.arange(0.0, 1.01 - w_lr, 0.1):
            w_xg = round(1.0 - w_lr - w_lg, 2)
            if w_xg < 0: continue
            p = w_lr * proba_lr_va + w_lg * proba_lgb_va + w_xg * proba_xgb_va
            yp_a = classes[p.argmax(1)]
            score = accuracy_score(y_va, yp_a)
            if score > best["metric"]:
                best = {"metric": score, "w_lr": round(w_lr, 2), "w_lg": round(w_lg, 2), "w_xg": w_xg, "proba": p}
    print(f"\nBest blend by accuracy: LR={best['w_lr']}  LGBM={best['w_lg']}  XGB={best['w_xg']}")
    r_blend_a = report("BLEND (argmax)",  y_va, classes[best["proba"].argmax(1)])
    r_blend_e = report("BLEND (E-round)", y_va, eround(best["proba"], classes))

    all_metrics = {
        "LR_argmax": r1a, "LR_eround": r1e,
        "LGBM_argmax": r2a, "LGBM_eround": r2e,
        "XGB_argmax": r3a, "XGB_eround": r3e,
        "BLEND_argmax": r_blend_a, "BLEND_eround": r_blend_e,
        "blend_weights": {"LR": best["w_lr"], "LGBM": best["w_lg"], "XGB": best["w_xg"]},
    }

    # ---- Refit on full train, predict test ----
    print("\n== Refit on full train ==")
    t0 = time.time()
    v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=200_000)
    Xw_tr = v_w.fit_transform(tr["text"]); Xw_te = v_w.transform(te["text"])
    Xc_tr = v_c.fit_transform(tr["text"]); Xc_te = v_c.transform(te["text"])
    X_tfidf_tr = hstack([Xw_tr, Xc_tr]).tocsr(); X_tfidf_te = hstack([Xw_te, Xc_te]).tocsr()
    print(f"  tfidf full: {time.time()-t0:.1f}s  shape={X_tfidf_tr.shape}")

    t0 = time.time()
    lr = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs", class_weight="balanced")
    lr.fit(X_tfidf_tr, tr["rate"])
    proba_lr_te = lr.predict_proba(X_tfidf_te)
    print(f"  LR: {time.time()-t0:.1f}s")

    t0 = time.time()
    svd = TruncatedSVD(n_components=200, random_state=RNG)
    Xsvd_tr = svd.fit_transform(X_tfidf_tr); Xsvd_te = svd.transform(X_tfidf_te)
    S_tr = stylo(tr["text"]).values; S_te = stylo(te["text"]).values
    X_dense_tr = np.hstack([Xsvd_tr, S_tr]); X_dense_te = np.hstack([Xsvd_te, S_te])
    print(f"  SVD+stylo: {time.time()-t0:.1f}s")

    t0 = time.time()
    lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, class_weight="balanced", random_state=RNG, verbose=-1)
    lgbm.fit(X_dense_tr, tr["rate"])
    proba_lgb_te = lgbm.predict_proba(X_dense_te)
    print(f"  LGBM: {time.time()-t0:.1f}s")

    t0 = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
        colsample_bytree=0.8, random_state=RNG, n_jobs=-1, eval_metric="mlogloss",
    )
    xgbm.fit(X_dense_tr, tr["rate"].values - 1)
    proba_xgb_te = xgbm.predict_proba(X_dense_te)
    print(f"  XGB: {time.time()-t0:.1f}s")

    # ---- Apply best blend ----
    p = best["w_lr"] * proba_lr_te + best["w_lg"] * proba_lgb_te + best["w_xg"] * proba_xgb_te
    pred_argmax = classes[p.argmax(1)]
    pred_eround = eround(p, classes)

    pd.DataFrame({"index": te["index"], "rate": pred_argmax.astype(int)}).to_csv(RES / "submission_best_argmax.csv", index=False)
    pd.DataFrame({"index": te["index"], "rate": pred_eround.astype(int)}).to_csv(RES / "submission_best_eround.csv", index=False)
    print("\nwrote results/submission_best_argmax.csv")
    print("wrote results/submission_best_eround.csv")
    print(f"\nFinal val metrics for the chosen blend (argmax):  {r_blend_a}")
    print(f"Final val metrics for the chosen blend (E-round): {r_blend_e}")

    (RES / "metrics_best.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print("wrote results/metrics_best.json")


if __name__ == "__main__":
    main()
