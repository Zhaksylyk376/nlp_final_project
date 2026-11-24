"""Fine-tune cointegrated/rubert-tiny2 on a train subsample.
Produces test probabilities and a submission; the probabilities are also
saved so we can stack with CatBoost / TF-IDF ensemble later.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

torch.set_num_threads(4)  # leave CPU for other running jobs

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LEN = 96
BATCH = 32
EPOCHS = 2
LR = 3e-5
SUBSAMPLE = 20000   # per epoch; CPU, so keep modest
RNG = 0


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = list(texts); self.labels = list(labels); self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0), "labels": torch.tensor(int(self.labels[i]), dtype=torch.long)}


def main():
    t0 = time.time()
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RNG)
    tri, vai = next(sss.split(tr["text"], tr["rate"]))
    tr_sub = tr.iloc[tri].reset_index(drop=True)
    va_sub = tr.iloc[vai].reset_index(drop=True)

    if SUBSAMPLE and SUBSAMPLE < len(tr_sub):
        tr_sub = tr_sub.sample(SUBSAMPLE, random_state=RNG).reset_index(drop=True)
    print(f"load: {time.time()-t0:.1f}s  train={len(tr_sub)} val={len(va_sub)} test={len(te)}")

    print("loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    tr_loader = DataLoader(DS(tr_sub["text"], tr_sub["rate"] - 1, tok, MAX_LEN), batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(DS(va_sub["text"], va_sub["rate"] - 1, tok, MAX_LEN), batch_size=BATCH * 2, num_workers=0)

    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(tr_loader) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        mdl.train()
        t0 = time.time()
        tot = 0.0; n = 0
        for batch in tr_loader:
            opt.zero_grad()
            out = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item() * batch["labels"].size(0); n += batch["labels"].size(0)
        print(f"epoch {epoch}: loss={tot/n:.4f}  time={time.time()-t0:.1f}s")

        # Validation
        mdl.eval()
        with torch.inference_mode():
            logits = []
            for batch in va_loader:
                logits.append(mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.cpu().numpy())
            logits = np.concatenate(logits)
            proba_va = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            yp = proba_va.argmax(1) + 1
            y = va_sub["rate"].values
            print(f"  val: acc={accuracy_score(y, yp):.4f}  mF1={f1_score(y, yp, average='macro'):.4f}  wF1={f1_score(y, yp, average='weighted'):.4f}  QWK={cohen_kappa_score(y, yp, weights='quadratic'):.4f}")

    # Final val proba (already have it) + test proba
    mdl.eval()
    with torch.inference_mode():
        te_loader = DataLoader(DS(te["text"], np.zeros(len(te), dtype=int), tok, MAX_LEN), batch_size=BATCH * 2)
        logits_te = []
        for batch in tqdm(te_loader, desc="test"):
            logits_te.append(mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.cpu().numpy())
        logits_te = np.concatenate(logits_te)
        proba_te = np.exp(logits_te) / np.exp(logits_te).sum(axis=1, keepdims=True)

    pred_te = proba_te.argmax(1) + 1
    pd.DataFrame({"index": te["index"], "rate": pred_te.astype(int)}).to_csv(RES / "submission_rubert.csv", index=False)
    print("wrote results/submission_rubert.csv")

    np.save(RES / "rubert_proba_val.npy", proba_va)
    np.save(RES / "rubert_proba_test.npy", proba_te)
    np.save(RES / "rubert_val_idx.npy", vai)

    y = va_sub["rate"].values; yp = proba_va.argmax(1) + 1
    (RES / "rubert_metrics.json").write_text(json.dumps({"acc": float(accuracy_score(y, yp)), "mF1": float(f1_score(y, yp, average="macro")), "wF1": float(f1_score(y, yp, average="weighted")), "qwk": float(cohen_kappa_score(y, yp, weights="quadratic"))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
