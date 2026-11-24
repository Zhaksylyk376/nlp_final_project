from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

NUM_CLASSES = 5


def tfidf_lr_pipeline(C: float = 2.0, max_features: int = 300_000, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    analyzer="word",
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=max_features,
                ),
            ),
            ("lr", LogisticRegression(C=C, max_iter=2000, n_jobs=-1, solver="saga", class_weight=class_weight)),
        ]
    )


def tfidf_char_lr_pipeline(C: float = 2.0, max_features: int = 300_000, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=max_features,
                ),
            ),
            ("lr", LogisticRegression(C=C, max_iter=2000, n_jobs=-1, solver="saga", class_weight=class_weight)),
        ]
    )


class _TextDS(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


@dataclass
class FTConfig:
    model_name: str = "cointegrated/rubert-tiny2"
    lr: float = 3e-5
    epochs: int = 3
    batch_size: int = 32
    max_len: int = 128
    weight_decay: float = 0.01
    use_class_weights: bool = True


def _class_weights(labels, n_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    inv = counts.sum() / (counts * n_classes + 1e-9)
    return torch.tensor(inv, dtype=torch.float32)


def fine_tune_cls(train_df, val_df=None, cfg: FTConfig = FTConfig()):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=NUM_CLASSES)
    cw = _class_weights(train_df["label"].values) if cfg.use_class_weights else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(mdl.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = DataLoader(
        _TextDS(train_df["text"], train_df["label"], tok, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=True,
    )
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(tr) * cfg.epochs), len(tr) * cfg.epochs)
    mdl.train()
    for epoch in range(cfg.epochs):
        for batch in tqdm(tr, desc=f"ep{epoch}"):
            opt.zero_grad()
            out = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            sched.step()
    mdl.eval()
    return mdl, tok


@torch.inference_mode()
def predict_proba_ft(mdl, tok, texts, max_len: int = 128, batch_size: int = 64) -> np.ndarray:
    probs = []
    texts = list(texts)
    for i in tqdm(range(0, len(texts), batch_size), desc="predict"):
        enc = tok(texts[i : i + batch_size], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        logits = mdl(**enc).logits
        probs.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(probs, axis=0)


def expected_rating(proba: np.ndarray) -> np.ndarray:
    """E[rate] over classes 1..5 — useful proxy to argmax when target is QWK/MAE."""
    return proba @ np.arange(1, NUM_CLASSES + 1)


def expected_round(proba: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(expected_rating(proba)).astype(int), 1, NUM_CLASSES)
