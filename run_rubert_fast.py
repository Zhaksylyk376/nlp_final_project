"""Fast fine-tune of cointegrated/rubert-tiny2.
Optimised for CPU: all threads, short seq, 1 epoch, parallel data loader.
Prints per-batch progress so user sees it moving.
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

torch.set_num_threads(torch.get_num_threads())
print(f"torch threads = {torch.get_num_threads()}")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LEN = 64
BATCH = 64
EPOCHS = 1
LR = 5e-5
SUBSAMPLE = 15000
RNG = 0


class DS(Dataset):
    def __init__(self, enc_ids, enc_mask, labels):
        self.ids = enc_ids; self.mask = enc_mask; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.labels[i]}


def tokenize_all(texts, tok):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


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
    print(f"load+split: {time.time()-t0:.1f}s  tr={len(tr_sub)} va={len(va_sub)} te={len(te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    tr_ids, tr_mask = tokenize_all(tr_sub["text"], tok)
    va_ids, va_mask = tokenize_all(va_sub["text"], tok)
    te_ids, te_mask = tokenize_all(te["text"], tok)
    print(f"tokenize all: {time.time()-t0:.1f}s", flush=True)

    tr_labels = torch.tensor(tr_sub["rate"].values - 1, dtype=torch.long)
    va_labels = torch.tensor(va_sub["rate"].values - 1, dtype=torch.long)
    tr_loader = DataLoader(DS(tr_ids, tr_mask, tr_labels), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(DS(va_ids, va_mask, va_labels), batch_size=BATCH * 2)

    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total = len(tr_loader) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        mdl.train()
        t_ep = time.time()
        running_loss = 0.0
        for i, batch in enumerate(tr_loader):
            opt.zero_grad()
            out = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sch.step()
            running_loss += loss.item()
            if (i + 1) % 20 == 0:
                avg = running_loss / 20
                elapsed = time.time() - t_ep
                print(f"  [epoch {epoch}] batch {i+1}/{len(tr_loader)}  avg_loss={avg:.4f}  elapsed={elapsed:.0f}s", flush=True)
                running_loss = 0.0

        print(f"epoch {epoch} done in {time.time()-t_ep:.1f}s", flush=True)
        mdl.eval()
        with torch.inference_mode():
            logits = []
            for batch in va_loader:
                logits.append(mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.cpu().numpy())
            logits = np.concatenate(logits)
            proba_va = np.exp(logits - logits.max(1, keepdims=True))
            proba_va = proba_va / proba_va.sum(1, keepdims=True)
            yp = proba_va.argmax(1) + 1
            y = va_sub["rate"].values
            print(f"  val: acc={accuracy_score(y, yp):.4f}  mF1={f1_score(y, yp, average='macro'):.4f}  wF1={f1_score(y, yp, average='weighted'):.4f}", flush=True)

    # Test prediction
    print("predict test...", flush=True)
    mdl.eval()
    te_loader = DataLoader(DS(te_ids, te_mask, torch.zeros(len(te), dtype=torch.long)), batch_size=BATCH * 2)
    with torch.inference_mode():
        logits_te = []
        for batch in te_loader:
            logits_te.append(mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.cpu().numpy())
        logits_te = np.concatenate(logits_te)
        proba_te = np.exp(logits_te - logits_te.max(1, keepdims=True))
        proba_te = proba_te / proba_te.sum(1, keepdims=True)

    pred_te = proba_te.argmax(1) + 1
    pd.DataFrame({"index": te["index"], "rate": pred_te.astype(int)}).to_csv(RES / "submission_rubert.csv", index=False)
    print("wrote results/submission_rubert.csv", flush=True)

    np.save(RES / "rubert_proba_val.npy", proba_va)
    np.save(RES / "rubert_proba_test.npy", proba_te)
    np.save(RES / "rubert_val_idx.npy", vai)
    y = va_sub["rate"].values; yp = proba_va.argmax(1) + 1
    (RES / "rubert_metrics.json").write_text(json.dumps({"acc": float(accuracy_score(y, yp)), "mF1": float(f1_score(y, yp, average="macro")), "wF1": float(f1_score(y, yp, average="weighted"))}, indent=2))


if __name__ == "__main__":
    main()
