"""Same ensemble but WITHOUT class_weight='balanced'.
If the leaderboard metric is weighted-F1 or accuracy, letting the model
predict the majority class more often typically improves the score by
3-6 pp relative to the balanced version.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

RNG = 0

_WORD = re.compile(r"[А-Яа-яЁё]+")
_SENT = re.compile(r"[.!?…]+")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
POS_CUE = ("хорош", "отличн", "супер", "вкусн", "рекоменд", "удобн", "спасиб", "вежлив", "прият", "быстр", "свеж")
NEG_CUE = ("плох", "ужас", "отврат", "грязн", "хам", "обман", "дорог", "никогда", "груб", "медленн", "просроч")


def stylo(texts):
    out = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        toks = [w.lower() for w in _WORD.findall(t)]
        n = max(len(toks), 1)
        sents = [s for s in _SENT.split(t) if s.strip()] or [t]
        sl = [len(_WORD.findall(s)) for s in sents] or [n]
        excl = t.count("!"); quest = t.count("?")
        emoji = len(_EMOJI.findall(t))
        pos = sum(t.lower().count(w) for w in POS_CUE)
        neg = sum(t.lower().count(w) for w in NEG_CUE)
        out.append([len(t), n, len(set(toks))/n, np.mean(sl), np.std(sl), excl/n, quest/n, emoji, pos, neg, pos-neg])
    return np.array(out, dtype=np.float32)


def main():
    t0 = time.time()
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")
    print(f"loaded in {time.time()-t0:.1f}s")

    # 10% stratified val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RNG)
    tri, vai = next(sss.split(tr["text"], tr["rate"]))
    tr_sub, va_sub = tr.iloc[tri].reset_index(drop=True), tr.iloc[vai].reset_index(drop=True)

    # --- TF-IDF features ---
    t0 = time.time()
    v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=200_000)
    Xw_tr = v_w.fit_transform(tr_sub["text"]); Xw_va = v_w.transform(va_sub["text"])
    Xc_tr = v_c.fit_transform(tr_sub["text"]); Xc_va = v_c.transform(va_sub["text"])
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr(); X_va = hstack([Xw_va, Xc_va]).tocsr()
    print(f"tfidf: {time.time()-t0:.1f}s  shape={X_tr.shape}")

    # --- Model 1: LR (NO class_weight) ---
    t0 = time.time()
    lr = LogisticRegression(C=4.0, max_iter=400, solver="lbfgs")
    lr.fit(X_tr, tr_sub["rate"])
    p_lr_va = lr.predict_proba(X_va)
    print(f"LR fit: {time.time()-t0:.1f}s")

    # --- SVD + stylo ---
    t0 = time.time()
    svd = TruncatedSVD(n_components=200, random_state=RNG)
    Xsvd_tr = svd.fit_transform(X_tr); Xsvd_va = svd.transform(X_va)
    S_tr = stylo(tr_sub["text"]); S_va = stylo(va_sub["text"])
    D_tr = np.hstack([Xsvd_tr, S_tr]); D_va = np.hstack([Xsvd_va, S_va])
    print(f"SVD+stylo: {time.time()-t0:.1f}s")

    # --- LGBM / XGB without class_weight ---
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, random_state=RNG, verbose=-1)
    lgbm.fit(D_tr, tr_sub["rate"])
    p_lg_va = lgbm.predict_proba(D_va)
    print(f"LGBM fit: {time.time()-t0:.1f}s")

    t0 = time.time()
    xgbm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=RNG, n_jobs=-1, eval_metric="mlogloss")
    xgbm.fit(D_tr, tr_sub["rate"].values - 1)
    p_xg_va = xgbm.predict_proba(D_va)
    print(f"XGB fit: {time.time()-t0:.1f}s")

    classes = lr.classes_
    y = va_sub["rate"].values

    def rep(name, yp):
        acc = accuracy_score(y, yp); mf = f1_score(y, yp, average="macro"); wf = f1_score(y, yp, average="weighted"); qk = cohen_kappa_score(y, yp, weights="quadratic")
        print(f"  {name:22s}  acc={acc:.4f}  mF1={mf:.4f}  wF1={wf:.4f}  QWK={qk:.4f}")
        return {"acc": acc, "mF1": mf, "wF1": wf, "qwk": qk}

    print("\n== Validation (no class_weight) ==")
    r_lr = rep("LR argmax", classes[p_lr_va.argmax(1)])
    r_lg = rep("LGBM argmax", classes[p_lg_va.argmax(1)])
    r_xg = rep("XGB argmax", classes[p_xg_va.argmax(1)])

    # Search blend by WEIGHTED F1 (closest to leaderboard F1 if dataset-level)
    best = {"wF1": -1}
    for w_lr in np.arange(0.0, 1.01, 0.1):
        for w_lg in np.arange(0.0, 1.01 - w_lr, 0.1):
            w_xg = round(1.0 - w_lr - w_lg, 2)
            if w_xg < 0: continue
            p = w_lr * p_lr_va + w_lg * p_lg_va + w_xg * p_xg_va
            yp = classes[p.argmax(1)]
            wf = f1_score(y, yp, average="weighted")
            if wf > best["wF1"]:
                best = {"wF1": wf, "w_lr": round(w_lr, 2), "w_lg": round(w_lg, 2), "w_xg": w_xg, "proba": p}
    print(f"\nBest blend by weighted-F1: LR={best['w_lr']}  LGBM={best['w_lg']}  XGB={best['w_xg']}")
    r_b = rep("BLEND argmax", classes[best["proba"].argmax(1)])

    # Also search by macro-F1 (in case leaderboard uses macro)
    best_m = {"mF1": -1}
    for w_lr in np.arange(0.0, 1.01, 0.1):
        for w_lg in np.arange(0.0, 1.01 - w_lr, 0.1):
            w_xg = round(1.0 - w_lr - w_lg, 2)
            if w_xg < 0: continue
            p = w_lr * p_lr_va + w_lg * p_lg_va + w_xg * p_xg_va
            yp = classes[p.argmax(1)]
            mf = f1_score(y, yp, average="macro")
            if mf > best_m["mF1"]:
                best_m = {"mF1": mf, "w_lr": round(w_lr, 2), "w_lg": round(w_lg, 2), "w_xg": w_xg, "proba": p}
    print(f"Best blend by macro-F1:    LR={best_m['w_lr']}  LGBM={best_m['w_lg']}  XGB={best_m['w_xg']}")
    r_bm = rep("BLEND macro-tuned", classes[best_m["proba"].argmax(1)])

    # ---- Refit on full train ----
    print("\n== Refit on full train ==")
    t0 = time.time()
    v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    v_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True, max_features=200_000)
    Xw_tr = v_w.fit_transform(tr["text"]); Xw_te = v_w.transform(te["text"])
    Xc_tr = v_c.fit_transform(tr["text"]); Xc_te = v_c.transform(te["text"])
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr(); X_te = hstack([Xw_te, Xc_te]).tocsr()

    lr = LogisticRegression(C=4.0, max_iter=400, solver="lbfgs"); lr.fit(X_tr, tr["rate"])
    p_lr_te = lr.predict_proba(X_te)
    print(f"  LR: {time.time()-t0:.1f}s"); t0 = time.time()

    svd = TruncatedSVD(n_components=200, random_state=RNG)
    Xsvd_tr = svd.fit_transform(X_tr); Xsvd_te = svd.transform(X_te)
    D_tr = np.hstack([Xsvd_tr, stylo(tr["text"])]); D_te = np.hstack([Xsvd_te, stylo(te["text"])])

    lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20, random_state=RNG, verbose=-1)
    lgbm.fit(D_tr, tr["rate"]); p_lg_te = lgbm.predict_proba(D_te)
    print(f"  LGBM: {time.time()-t0:.1f}s"); t0 = time.time()

    xgbm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8, random_state=RNG, n_jobs=-1, eval_metric="mlogloss")
    xgbm.fit(D_tr, tr["rate"].values - 1); p_xg_te = xgbm.predict_proba(D_te)
    print(f"  XGB: {time.time()-t0:.1f}s")

    # Apply both blends
    p_wf = best["w_lr"] * p_lr_te + best["w_lg"] * p_lg_te + best["w_xg"] * p_xg_te
    p_mf = best_m["w_lr"] * p_lr_te + best_m["w_lg"] * p_lg_te + best_m["w_xg"] * p_xg_te
    pd.DataFrame({"index": te["index"], "rate": classes[p_wf.argmax(1)].astype(int)}).to_csv(RES / "submission_unbal_wF1.csv", index=False)
    pd.DataFrame({"index": te["index"], "rate": classes[p_mf.argmax(1)].astype(int)}).to_csv(RES / "submission_unbal_mF1.csv", index=False)
    # Also save single-model LR as a safety net
    pd.DataFrame({"index": te["index"], "rate": classes[p_lr_te.argmax(1)].astype(int)}).to_csv(RES / "submission_unbal_LR.csv", index=False)
    print("\nwrote results/submission_unbal_wF1.csv  (tuned for weighted-F1)")
    print("wrote results/submission_unbal_mF1.csv  (tuned for macro-F1)")
    print("wrote results/submission_unbal_LR.csv   (single LR, no class_weight)")

    summary = {"LR": r_lr, "LGBM": r_lg, "XGB": r_xg, "BLEND_wF1": r_b, "BLEND_mF1": r_bm, "blend_wF1_weights": {"LR": best["w_lr"], "LGBM": best["w_lg"], "XGB": best["w_xg"]}, "blend_mF1_weights": {"LR": best_m["w_lr"], "LGBM": best_m["w_lg"], "XGB": best_m["w_xg"]}}
    (RES / "metrics_unbalanced.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
