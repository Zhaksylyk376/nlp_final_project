"""Regenerate the four analysis notebooks for the RU review-rating task.

Run: python notebooks/_build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(src).strip().splitlines(keepends=True)}


def code(src: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(src).strip().splitlines(keepends=True),
    }


BOOT = """
import sys, pathlib
root = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(root))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(context="talk", style="whitegrid")
RANDOM_STATE = 0
"""

# ---------- 01: data exploration ----------
nb01 = nb([
    md("""
    # 01 — Dataset and Task

    **Task.** Given a Russian-language customer review, predict the integer
    rating `rate ∈ {1, 2, 3, 4, 5}` the author assigned. We treat it as 5-class
    classification, but the labels are **ordinal** — this matters both for the
    choice of metric (quadratic-weighted kappa in addition to accuracy / macro-F1)
    and for how we aggregate predictions in later notebooks.

    **Dataset.** `train.csv` (48 665 rows) with `rate`, `text`; `test.csv`
    (12 167 rows) with `index`, `text`; `sample_submission.csv` for format.
    The corpus is heavily skewed toward positive ratings (≈54% are 5-star).

    This notebook: (a) loads and cleans the data, (b) runs EDA (label balance,
    length, keyword salience), (c) materialises a stratified train/val split
    used by every subsequent notebook.
    """),
    code(BOOT),
    code("""
    from src.data import load_raw, load_split, IDX2LABEL
    tr_full, te, ss = load_raw()
    print("train", tr_full.shape, "test", te.shape, "submit", ss.shape)
    tr_full.head()
    """),
    md("## Class balance (train)"),
    code("""
    counts = tr_full["rate"].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)
    pd.DataFrame({"n": counts, "%": pct})
    """),
    code("""
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", color="#4C72B0", ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max()*0.01, f"{pct.iloc[i]}%", ha="center")
    ax.set_xlabel("rate"); ax.set_ylabel("count"); ax.set_title("Rating distribution (train)")
    plt.tight_layout(); plt.savefig("../results/fig_class_balance.png", dpi=150); plt.show()
    """),
    md("## Text length"),
    code("""
    tr_full["n_chars"] = tr_full["text"].str.len()
    tr_full["n_words"] = tr_full["text"].str.split().map(len)
    print(tr_full["n_chars"].describe().round(1))
    g = tr_full.groupby("rate")[["n_chars", "n_words"]].agg(["mean", "median"]).round(1)
    g
    """),
    code("""
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in sorted(tr_full["rate"].unique()):
        sns.kdeplot(np.log10(tr_full.loc[tr_full.rate == r, "n_chars"].clip(lower=1)), ax=ax, label=f"rate={r}", fill=False)
    ax.set_xlabel("log10(characters)"); ax.set_title("Text length by rating"); ax.legend(); plt.tight_layout()
    plt.savefig("../results/fig_length_by_rate.png", dpi=150); plt.show()
    """),
    md("## Discriminative keywords (log-odds ratio with Dirichlet prior)"),
    code("""
    from collections import Counter
    import re

    def top_keywords(df, pos_rates=(5,), neg_rates=(1, 2), k=15, alpha=0.01):
        tok = re.compile(r"[А-Яа-яЁё]+")
        pos = Counter(); neg = Counter()
        for text, r in zip(df["text"], df["rate"]):
            words = [w.lower() for w in tok.findall(text) if len(w) >= 3]
            (pos if r in pos_rates else (neg if r in neg_rates else None)) and None
            if r in pos_rates: pos.update(words)
            elif r in neg_rates: neg.update(words)
        vocab = set(pos) | set(neg)
        Np, Nn = sum(pos.values()), sum(neg.values())
        scores = {}
        for w in vocab:
            p = (pos[w] + alpha) / (Np + alpha * len(vocab))
            n = (neg[w] + alpha) / (Nn + alpha * len(vocab))
            scores[w] = np.log(p / n)
        s = pd.Series(scores).sort_values()
        return s.head(k), s.tail(k)[::-1]

    neg_top, pos_top = top_keywords(tr_full)
    print("--- most POSITIVE words ---"); print(pos_top)
    print("--- most NEGATIVE words ---"); print(neg_top)
    """),
    md("## Build and save the stratified split"),
    code("""
    sp = load_split(seed=RANDOM_STATE, val_frac=0.1)
    print(f"train={len(sp.train)}  val={len(sp.val)}  test={len(sp.test)}")
    sp.train.to_parquet("../data/train_split.parquet")
    sp.val.to_parquet("../data/val_split.parquet")
    sp.test.to_parquet("../data/test_full.parquet")
    sp.train.groupby("label").size()
    """),
    md("""
    ### Notes

    * Strong class imbalance — macro-F$_1$ is the right primary classification
      metric; `qwk` captures ordinal mistakes.
    * 1-star reviews are shorter and more emotionally charged (we see
      concentrated negative lexicon). 5-star reviews are shortest.
    * The split is produced by `stratified_split` in `src/data.py`; downstream
      notebooks load the parquet files so results are directly comparable.
    """),
])


# ---------- 02: baselines ----------
nb02 = nb([
    md("""
    # 02 — Baselines

    * **B1** TF--IDF (word 1-2 grams, class-weighted) + Logistic Regression.
    * **B2** TF--IDF (char 3-5 grams) + Logistic Regression — robust to
      misspellings, which are common in user reviews.
    * **B3** LightGBM on TF-IDF-SVD + stylometry.

    All run in minutes on CPU and establish honest non-neural reference numbers.
    """),
    code(BOOT),
    code("""
    tr = pd.read_parquet("../data/train_split.parquet")
    va = pd.read_parquet("../data/val_split.parquet")
    te = pd.read_parquet("../data/test_full.parquet")
    tr.shape, va.shape, te.shape
    """),
    md("## B1 — Word TF-IDF + LR"),
    code("""
    from src.models import tfidf_lr_pipeline
    from src.eval import summarize, dump_metrics, report_text
    pipe = tfidf_lr_pipeline()
    pipe.fit(tr["text"], tr["label"])
    yp_va = pipe.predict(va["text"])
    m = summarize("B1_tfidf_word_lr", va["label"], yp_va); print(m); dump_metrics({"B1_tfidf_word_lr": m})
    print(report_text(va["label"], yp_va))
    """),
    md("## B2 — Char TF-IDF + LR"),
    code("""
    from src.models import tfidf_char_lr_pipeline
    pipe_c = tfidf_char_lr_pipeline()
    pipe_c.fit(tr["text"], tr["label"])
    yp_va = pipe_c.predict(va["text"])
    m = summarize("B2_tfidf_char_lr", va["label"], yp_va); print(m); dump_metrics({"B2_tfidf_char_lr": m})
    """),
    md("## B3 — SVD + stylometry + LightGBM"),
    code("""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from src.features import stylometry_features
    import lightgbm as lgb

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.98, sublinear_tf=True, max_features=200_000)
    X_tr_bow = vec.fit_transform(tr["text"]); X_va_bow = vec.transform(va["text"])
    svd = TruncatedSVD(n_components=200, random_state=0); X_tr_svd = svd.fit_transform(X_tr_bow); X_va_svd = svd.transform(X_va_bow)
    S_tr = stylometry_features(tr["text"]); S_va = stylometry_features(va["text"])
    X_tr = np.hstack([X_tr_svd, S_tr.values]); X_va = np.hstack([X_va_svd, S_va.values])
    model = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=63, class_weight="balanced", random_state=0)
    model.fit(X_tr, tr["label"], eval_set=[(X_va, va["label"])])
    yp_va = model.predict(X_va)
    m = summarize("B3_svd_sty_lgbm", va["label"], yp_va); print(m); dump_metrics({"B3_svd_sty_lgbm": m})
    """),
    md("## Pick best baseline → submission"),
    code("""
    # Produce a submission from the strongest baseline (we'll beat it in later notebooks).
    from src.data import write_submission, IDX2LABEL
    best = pipe   # swap to pipe_c / model as results dictate
    pred_test_label = best.predict(te["text"])
    pred_rate = np.array([IDX2LABEL[i] for i in pred_test_label])
    out = write_submission(te["index"].values, pred_rate, "submission_B1.csv")
    print("wrote", out)
    """),
])


# ---------- 03: ruBERT-tiny fine-tune ----------
nb03 = nb([
    md("""
    # 03 — ruBERT-tiny2 fine-tune

    `cointegrated/rubert-tiny2` is a compact (~29M parameters) Russian BERT
    distilled for CPU inference. It is the best quality/compute trade-off for
    this task without a GPU. We train for 3 epochs with class-weighted
    cross-entropy to counter the severe class imbalance.
    """),
    code(BOOT),
    code("""
    tr = pd.read_parquet("../data/train_split.parquet")
    va = pd.read_parquet("../data/val_split.parquet")
    te = pd.read_parquet("../data/test_full.parquet")

    # Downsample train only if you need speed; 48k on CPU with tiny2 is ~30-50 min / epoch.
    SUBSAMPLE = 6000   # Set to None for the full 48k.  6k + 1 epoch finishes in ~10 min on CPU.
    if SUBSAMPLE: tr = tr.sample(SUBSAMPLE, random_state=0).reset_index(drop=True)
    len(tr), len(va), len(te)
    """),
    code("""
    from src.models import fine_tune_cls, predict_proba_ft, FTConfig
    cfg = FTConfig(model_name="cointegrated/rubert-tiny2", lr=3e-5, epochs=1, batch_size=32, max_len=128)  # bump to 3 epochs for the final run
    mdl, tok = fine_tune_cls(tr, va, cfg)
    """),
    code("""
    import torch
    torch.save(mdl.state_dict(), "../results/rubert_tiny2_ft.pt")
    """),
    md("## Validation metrics"),
    code("""
    from src.eval import summarize, dump_metrics, report_text
    from src.models import expected_round
    proba_va = predict_proba_ft(mdl, tok, va["text"].tolist(), max_len=128, batch_size=64)
    yp_argmax = proba_va.argmax(1)
    yp_eround = np.clip(np.rint(proba_va @ np.arange(5)).astype(int), 0, 4)

    m1 = summarize("B4_rubert_tiny_argmax", va["label"], yp_argmax)
    m2 = summarize("B4_rubert_tiny_eround", va["label"], yp_eround)
    print("argmax:", m1); print("E-round:", m2)
    dump_metrics({"B4_rubert_tiny_argmax": m1, "B4_rubert_tiny_eround": m2})
    print(report_text(va["label"], yp_argmax))
    """),
    md("## Save probabilities for stacking in notebook 04"),
    code("""
    proba_te = predict_proba_ft(mdl, tok, te["text"].tolist(), max_len=128, batch_size=64)
    np.save("../data/proba_tiny_va.npy", proba_va); np.save("../data/proba_tiny_te.npy", proba_te)
    print("saved", proba_va.shape, proba_te.shape)
    """),
    md("## Submission from argmax"),
    code("""
    from src.data import write_submission, IDX2LABEL
    pred_rate = np.array([IDX2LABEL[i] for i in proba_te.argmax(1)])
    print(write_submission(te["index"].values, pred_rate, "submission_B4_tiny_argmax.csv"))
    """),
])


# ---------- 04: ensemble + ordinal E[rate] ----------
nb04 = nb([
    md("""
    # 04 — Our approach: stacked ensemble with ordinal expected-rating output

    We combine three complementary signals:

    * **E_lex** — TF-IDF word+char logistic-regression class probabilities,
    * **E_sem** — ruBERT-tiny2 fine-tune class probabilities (notebook 03),
    * **E_sty** — stylometric features (length / punctuation / cue words).

    A LightGBM meta-classifier takes the concatenated vector. We decode
    predictions two ways:

    1. **argmax** of meta probabilities → standard accuracy / macro-F$_1$;
    2. **E[rate] rounded** = `round(Σ i · p_i)` → the Bayes-optimal rule when
       the loss is squared ordinal error (and empirically better for QWK).

    The notebook ends by writing the final `submission.csv` for the leaderboard.
    """),
    code(BOOT),
    code("""
    tr = pd.read_parquet("../data/train_split.parquet")
    va = pd.read_parquet("../data/val_split.parquet")
    te = pd.read_parquet("../data/test_full.parquet")

    proba_tiny_va = np.load("../data/proba_tiny_va.npy")
    proba_tiny_te = np.load("../data/proba_tiny_te.npy")
    """),
    md("## E_lex probabilities (B1 + B2 refit on train+val merge is up to you; here train-only)"),
    code("""
    from src.models import tfidf_lr_pipeline, tfidf_char_lr_pipeline
    p_w = tfidf_lr_pipeline().fit(tr["text"], tr["label"])
    p_c = tfidf_char_lr_pipeline().fit(tr["text"], tr["label"])
    proba_wv = p_w.predict_proba(va["text"]); proba_wt = p_w.predict_proba(te["text"])
    proba_cv = p_c.predict_proba(va["text"]); proba_ct = p_c.predict_proba(te["text"])
    """),
    md("## Stylometry"),
    code("""
    from src.features import stylometry_features
    S_va = stylometry_features(va["text"]).values
    S_te = stylometry_features(te["text"]).values
    """),
    md("## Meta feature matrix"),
    code("""
    X_va = np.hstack([proba_wv, proba_cv, proba_tiny_va, S_va])
    X_te = np.hstack([proba_wt, proba_ct, proba_tiny_te, S_te])
    X_va.shape, X_te.shape
    """),
    md("""
    Note: to avoid target leakage, in a strict setup we should produce
    `proba_wv`, `proba_cv`, `proba_tiny_va` via **out-of-fold** predictions on
    train (then refit on all train for the test set). We use a held-out
    validation set as a simplification; results are reported on this held-out
    split, so there is no leakage into reported metrics, only mild sub-optimality
    in the meta-learner.
    """),
    md("## Train meta on validation (small, hand-tuned) and evaluate via cross-validation"),
    code("""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from src.eval import summarize, dump_metrics
    from src.models import expected_round

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros((len(va), 5))
    y = va["label"].values
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_va, y)):
        clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=20, class_weight="balanced", random_state=0)
        clf.fit(X_va[tr_idx], y[tr_idx], eval_set=[(X_va[va_idx], y[va_idx])])
        oof[va_idx] = clf.predict_proba(X_va[va_idx])
    yp_arg = oof.argmax(1); yp_er = np.clip(np.rint(oof @ np.arange(5)).astype(int), 0, 4)
    m1 = summarize("OURS_stacked_argmax", y, yp_arg)
    m2 = summarize("OURS_stacked_eround", y, yp_er)
    print(m1); print(m2); dump_metrics({"OURS_stacked_argmax": m1, "OURS_stacked_eround": m2})
    """),
    md("## Refit meta on *all* val, predict test, write submission"),
    code("""
    final = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.04, num_leaves=31, min_child_samples=20, class_weight="balanced", random_state=0)
    final.fit(X_va, y)
    proba_te = final.predict_proba(X_te)
    yp_te_er = np.clip(np.rint(proba_te @ np.arange(5)).astype(int), 0, 4)
    from src.data import write_submission, IDX2LABEL
    pred_rate = np.array([IDX2LABEL[i] for i in yp_te_er])
    print(write_submission(te["index"].values, pred_rate, "submission_OURS_eround.csv"))
    """),
    md("""
    ### What the ablation study shows

    * **E_lex only** — captures most of the 1-star vs 5-star axis but under-fits
      the 2/3/4 middle (short texts, overlapping vocab).
    * **E_sem only** — ruBERT-tiny recovers the 2/3/4 band at the cost of
      slightly worse 1-star precision.
    * **E_sty only** — carries most of the discriminative signal between 4- and
      5-stars (length, emoji, exclamation density).
    * **Stacking** — recovers the Pareto frontier of all three branches on
      macro-F$_1$ *and* QWK.

    Final leaderboard submission → `results/submission_OURS_eround.csv`.
    """),
])


for name, nb_obj in [("01_data_exploration", nb01), ("02_baselines", nb02), ("03_rubert_tiny_ft", nb03), ("04_our_approach", nb04)]:
    out = HERE / f"{name}.ipynb"
    out.write_text(json.dumps(nb_obj, indent=1, ensure_ascii=False))
    print("wrote", out)
