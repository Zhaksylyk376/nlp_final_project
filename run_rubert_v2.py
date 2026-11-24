"""Second ruBERT run: more data, 2 epochs, max_len=96, no class_weight.
Goal: push val weighted-F1 to 0.65+ to beat the easy leaderboard baseline.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MODEL_NAME = "cointegrated/rubert-tiny2"
MAX_LEN = 96
BATCH = 48
EPOCHS = 2
LR = 3e-5
SUBSAMPLE = 25000
RNG = 0


class DS(Dataset):
    def __init__(self, ids, mask, labels):
        self.ids = ids; self.mask = mask; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.labels[i]}


def main():
    t0 = time.time()
    tr = pd.read_csv(RAW / "train.csv")
    te = pd.read_csv(RAW / "test.csv")
    tr["text"] = tr["text"].fillna(""); te["text"] = te["text"].fillna("")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RNG)
    tri, vai = next(sss.split(tr["text"], tr["rate"]))
    tr_sub = tr.iloc[tri].reset_index(drop=True)
    va_sub = tr.iloc[vai].reset_index(drop=True)
    if SUBSAMPLE < len(tr_sub):
        tr_sub = tr_sub.sample(SUBSAMPLE, random_state=RNG).reset_index(drop=True)
    print(f"loaded: tr={len(tr_sub)} va={len(va_sub)} te={len(te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    tr_e = tok(list(tr_sub["text"]), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    va_e = tok(list(va_sub["text"]), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    te_e = tok(list(te["text"]), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    print(f"tokenize: {time.time()-t0:.1f}s", flush=True)

    tr_y = torch.tensor(tr_sub["rate"].values - 1, dtype=torch.long)
    va_y = torch.tensor(va_sub["rate"].values - 1, dtype=torch.long)

    tr_ld = DataLoader(DS(tr_e["input_ids"], tr_e["attention_mask"], tr_y), batch_size=BATCH, shuffle=True)
    va_ld = DataLoader(DS(va_e["input_ids"], va_e["attention_mask"], va_y), batch_size=BATCH * 2)
    te_ld = DataLoader(DS(te_e["input_ids"], te_e["attention_mask"], torch.zeros(len(te), dtype=torch.long)), batch_size=BATCH * 2)

    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    total = len(tr_ld) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"total batches per epoch: {len(tr_ld)}", flush=True)
    for epoch in range(EPOCHS):
        mdl.train()
        t_ep = time.time()
        running = 0.0
        for i, batch in enumerate(tr_ld):
            opt.zero_grad()
            out = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sch.step()
            running += loss.item()
            if (i + 1) % 30 == 0:
                avg = running / 30
                el = time.time() - t_ep
                total_s = el / (i+1) * len(tr_ld)
                print(f"  ep{epoch} batch {i+1}/{len(tr_ld)}  loss={avg:.4f}  elapsed={el:.0f}s  epoch_eta={total_s:.0f}s", flush=True)
                running = 0.0

        print(f"epoch {epoch} done in {time.time()-t_ep:.1f}s", flush=True)
        mdl.eval()
        with torch.inference_mode():
            lg = []
            for b in va_ld:
                lg.append(mdl(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.cpu().numpy())
            lg = np.concatenate(lg)
            p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            yp = p.argmax(1) + 1
            y = va_sub["rate"].values
            acc = accuracy_score(y, yp); mF1 = f1_score(y, yp, average='macro'); wF1 = f1_score(y, yp, average='weighted')
            print(f"  val: acc={acc:.4f}  mF1={mF1:.4f}  wF1={wF1:.4f}", flush=True)

    # final test
    mdl.eval()
    with torch.inference_mode():
        lg = []
        for b in te_ld:
            lg.append(mdl(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.cpu().numpy())
        lg = np.concatenate(lg)
        proba_te = np.exp(lg - lg.max(1, keepdims=True)); proba_te /= proba_te.sum(1, keepdims=True)

    proba_va = p
    pred_te = proba_te.argmax(1) + 1
    pd.DataFrame({"index": te["index"], "rate": pred_te.astype(int)}).to_csv(RES / "submission_rubert_v2.csv", index=False)
    print("wrote results/submission_rubert_v2.csv", flush=True)
    np.save(RES / "rubert_v2_proba_val.npy", proba_va)
    np.save(RES / "rubert_v2_proba_test.npy", proba_te)
    (RES / "rubert_v2_metrics.json").write_text(json.dumps({"acc": float(acc), "mF1": float(mF1), "wF1": float(wF1)}, indent=2))


if __name__ == "__main__":
    main()
