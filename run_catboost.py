"""CatBoost with native text_features. Often the single strongest CPU model
on Russian short-review tasks because it jointly handles tokenization,
character n-grams, and BM25 features internally.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

RNG = 0


def main():
    t0 = time.time()
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RNG)
    tri, vai = next(sss.split(tr["text"], tr["rate"]))
    tr_sub, va_sub = tr.iloc[tri].reset_index(drop=True), tr.iloc[vai].reset_index(drop=True)
    print(f"split in {time.time()-t0:.1f}s  tr={len(tr_sub)} va={len(va_sub)}")

    tr_pool = Pool(tr_sub[["text"]], tr_sub["rate"] - 1, text_features=["text"])
    va_pool = Pool(va_sub[["text"]], va_sub["rate"] - 1, text_features=["text"])
    te_pool = Pool(te[["text"]], text_features=["text"])

    cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=RNG,
        early_stopping_rounds=100,
        task_type="CPU",
        thread_count=-1,
        text_processing={
            "tokenizers": [{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}],
            "dictionaries": [
                {"dictionary_id": "Word", "token_level_type": "Word", "occurrence_lower_bound": "3"},
                {"dictionary_id": "BiGram", "token_level_type": "Word", "gram_order": "2", "occurrence_lower_bound": "3"},
                {"dictionary_id": "Char4", "token_level_type": "Letter", "gram_order": "4", "occurrence_lower_bound": "5"},
            ],
            "feature_processing": {
                "default": [
                    {"tokenizers_names": ["Space"], "dictionaries_names": ["Word"], "feature_calcers": ["BoW", "NaiveBayes", "BM25"]},
                    {"tokenizers_names": ["Space"], "dictionaries_names": ["BiGram"], "feature_calcers": ["BoW"]},
                    {"tokenizers_names": ["Space"], "dictionaries_names": ["Char4"], "feature_calcers": ["BoW"]},
                ]
            },
        },
        verbose=100,
    )
    t0 = time.time()
    cb.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    print(f"\ntrain time: {time.time()-t0:.1f}s")

    y = va_sub["rate"].values
    proba_va = cb.predict_proba(va_pool)
    yp_va = cb.classes_[proba_va.argmax(1)] + 1  # back to 1..5
    acc = accuracy_score(y, yp_va); mf = f1_score(y, yp_va, average="macro"); wf = f1_score(y, yp_va, average="weighted"); qk = cohen_kappa_score(y, yp_va, weights="quadratic")
    print(f"CB val: acc={acc:.4f}  mF1={mf:.4f}  wF1={wf:.4f}  QWK={qk:.4f}")

    # Refit on full train
    t0 = time.time()
    full_pool = Pool(tr[["text"]], tr["rate"] - 1, text_features=["text"])
    cb_full = CatBoostClassifier(
        iterations=cb.tree_count_,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        random_seed=RNG,
        task_type="CPU",
        thread_count=-1,
        text_processing=cb.get_all_params()["text_processing"],
        verbose=200,
    )
    cb_full.fit(full_pool)
    print(f"refit time: {time.time()-t0:.1f}s  trees={cb.tree_count_}")

    proba_te = cb_full.predict_proba(te_pool)
    pred_te = cb_full.classes_[proba_te.argmax(1)] + 1
    pd.DataFrame({"index": te["index"], "rate": pred_te.astype(int)}).to_csv(RES / "submission_catboost.csv", index=False)
    print("wrote results/submission_catboost.csv")

    np.save(RES / "catboost_proba_val.npy", proba_va)
    np.save(RES / "catboost_proba_test.npy", proba_te)
    np.save(RES / "catboost_val_idx.npy", vai)
    (RES / "catboost_metrics.json").write_text(json.dumps({"acc": acc, "mF1": mf, "wF1": wf, "qwk": qk, "trees": int(cb.tree_count_)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
