# Russian Review Rating Prediction (NLP course final, Autumn 2026)

Five-class ordinal classification: predict the integer rating (1--5) a user assigned to a short Russian-language review. Dataset: 48 665 train rows, 12 167 test rows (format `index, text`), submission format `index, rate`.

## Repository layout

```
nlp_final_project/
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA, class balance, keywords, split
│   ├── 02_baselines.ipynb           # Word & char TF-IDF + LR, SVD + stylometry + LGBM
│   ├── 03_rubert_tiny_ft.ipynb      # rubert-tiny2 fine-tune (CPU)
│   └── 04_our_approach.ipynb        # Stacked ensemble + E-rating decoding + submission
├── src/
│   ├── data.py                      # loaders, stratified split, submission writer
│   ├── features.py                  # stylometry, embeddings (rubert-tiny2)
│   ├── models.py                    # pipelines, fine-tune wrapper, E-round decoder
│   └── eval.py                      # accuracy, macro-F1, QWK, MAE, dump_metrics
├── report/
│   ├── report.tex                   # ready for Overleaf
│   └── references.bib
├── data/
│   └── raw/                         # train.csv / test.csv / sample_submission.csv
├── results/                         # metrics.json + submission_*.csv
└── requirements.txt
```

## Setup

Python 3.14 breaks `transformers` + `sentencepiece`; pin 3.12. CPU is enough.

```fish
cd ~/nlp_final_project
uv venv .venv --python 3.12
source .venv/bin/activate.fish
uv pip install -r requirements.txt
jupyter lab
```

Run notebooks `01 → 04` in order. Numbers accumulate in `results/metrics.json`; final submission in `results/submission_OURS_eround.csv`.

## Task

* **Input:** short Russian review (median 86 chars, up to 4 kB).
* **Target:** integer `rate ∈ {1, 2, 3, 4, 5}` (ordinal, highly imbalanced — 54% are 5-star).
* **Primary metric (leaderboard):** accuracy on the hidden test split.
* **Secondary metrics we track:** macro-F$_1$, quadratic-weighted $\kappa$, MAE.

## Approach

| ID       | Features                                        | Model          | Role                        |
|----------|-------------------------------------------------|----------------|-----------------------------|
| B1       | Word TF–IDF (1–2 grams, min_df=2)               | LogReg (balanced) | Non-neural reference        |
| B2       | Char TF–IDF (3–5 grams)                         | LogReg (balanced) | Robust to misspellings      |
| B3       | TF-IDF-SVD (200-d) + 13 stylometry features     | LightGBM       | Non-neural upper bound      |
| B4       | `cointegrated/rubert-tiny2` fine-tuned 3 epochs | —              | Semantic branch             |
| **Ours** | $[p_\text{word}; p_\text{char}; p_\text{rubert}; \text{Sty}] \in \mathbb{R}^{28}$ | LightGBM meta + E-round decoding | Stacked ensemble            |

Decoding uses the Bayes-optimal rule under squared ordinal loss:
$\hat y = \mathrm{round}(\sum_i (i{+}1)\,p_i)$, clipped to $[1, 5]$ — this improves QWK/MAE at essentially no cost to accuracy.

## Previous art targets

* RuReviews 3-class sentiment: fastText + CNN → macro-F$_1$ 71% (Smetanin & Komarov 2019); transformer baselines reach ~83% (Zmitrovich et al. 2023). Our 5-class task is harder, but the relative gain of stacking matches their reported ensemble gains.
* Standard ordinal E-round decoding — Pedregosa et al. 2017 — is Bayes-optimal under squared loss; empirically improves QWK on all our models.

## Submitting

Running `04_our_approach.ipynb` writes `results/submission_OURS_eround.csv` in the required `index,rate` format. Upload to the competition page, paste the GitHub URL in the course submission's *Решение* field.

## License

Code: MIT. Dataset retains its competition license.
