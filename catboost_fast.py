"""Fast CatBoost — no eval split, no best-model tracking, just fit and predict."""
import time
import pandas as pd
from catboost import CatBoostClassifier, Pool

t0 = time.time()
tr = pd.read_csv("data/raw/train.csv")
te = pd.read_csv("data/raw/test.csv")
tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")

tr_pool = Pool(tr[["text"]], tr["rate"] - 1, text_features=["text"])
te_pool = Pool(te[["text"]], text_features=["text"])

cb = CatBoostClassifier(
    iterations=400,
    learning_rate=0.1,
    depth=6,
    loss_function="MultiClass",
    random_seed=0,
    thread_count=-1,
    verbose=50,
)
cb.fit(tr_pool)
print(f"total train: {time.time()-t0:.1f}s  trees={cb.tree_count_}")

pred = cb.predict(te_pool).flatten() + 1
pd.DataFrame({"index": te["index"], "rate": pred.astype(int)}).to_csv("results/submission_catboost_fast.csv", index=False)
print("wrote results/submission_catboost_fast.csv")
print(pd.Series(pred).value_counts().sort_index())
