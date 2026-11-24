from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

RU_FUNC = set(
    """и в во не на с со а по от у но из к о до за же со бы же ли или ни то что чтобы если бы же вот тут
    там здесь так как кто чей чья чьё чьи я ты он она оно мы вы они меня тебя его её нас вас их мне тебе
    ему ей нам вам им""".split()
)
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+")
_SENT = re.compile(r"[.!?…]+")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


def _tokens(text: str):
    return [w.lower() for w in _WORD.findall(text)]


def stylometry_features(texts: Iterable[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        toks = _tokens(text)
        n = max(len(toks), 1)
        sents = [s for s in _SENT.split(text) if s.strip()] or [text]
        sent_lens = [len(_tokens(s)) for s in sents] or [n]
        func = sum(1 for t in toks if t in RU_FUNC)
        upper = sum(1 for c in text if c.isupper())
        excl = text.count("!")
        quest = text.count("?")
        emoji = len(_EMOJI.findall(text))
        pos_cue = sum(text.lower().count(w) for w in ("хорош", "отличн", "супер", "вкусн", "рекоменд", "удобн", "любл", "спасиб"))
        neg_cue = sum(text.lower().count(w) for w in ("плох", "ужас", "отврат", "грязн", "хам", "обман", "дорог", "никогда", "ненавиж"))
        rows.append(
            {
                "n_chars": len(text),
                "n_tokens": n,
                "ttr": len(set(toks)) / n,
                "avg_sent_len": float(np.mean(sent_lens)),
                "std_sent_len": float(np.std(sent_lens)),
                "func_ratio": func / n,
                "upper_ratio": upper / max(len(text), 1),
                "excl_per_tok": excl / n,
                "quest_per_tok": quest / n,
                "emoji": emoji,
                "pos_cue": pos_cue,
                "neg_cue": neg_cue,
                "net_cue": pos_cue - neg_cue,
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def _enc(model_name: str = "cointegrated/rubert-tiny2"):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl


@torch.inference_mode()
def embed_texts(texts: Iterable[str], model_name: str = "cointegrated/rubert-tiny2", max_len: int = 128, batch_size: int = 32) -> np.ndarray:
    tok, mdl = _enc(model_name)
    texts = list(texts)
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"emb[{model_name}]"):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        h = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.numpy())
    return np.concatenate(out, axis=0)
