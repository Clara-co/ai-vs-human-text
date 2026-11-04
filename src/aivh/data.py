"""Load the Kaggle "AI Vs Human Text" corpus and draw balanced samples.

Dataset: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
File:    AI_Human.csv, columns `text` and `generated` (0 = human, 1 = AI),
         about 487k essays, roughly 305k human and 181k AI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "AI_Human.csv"
LABELS = {0: "human", 1: "ai"}


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. Run scripts/download_data.sh first (see README).")
    df = pd.read_csv(path)
    df = df.rename(columns={"generated": "label"})
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["text"]).drop_duplicates(subset="text")
    return df[["text", "label"]].reset_index(drop=True)


def balanced_split(
    df: pd.DataFrame, n_train: int, n_test: int, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disjoint train and test sets, each with an equal number of human and AI texts.

    The full corpus is ~63% human. A balanced sample keeps accuracy meaningful,
    because a model that always guesses "human" scores 50%, not 63%.
    """
    per_class = (n_train + n_test) // 2
    train_parts, test_parts = [], []
    for label, group in df.groupby("label"):
        if len(group) < per_class:
            raise ValueError(f"Only {len(group)} texts with label {label}; need {per_class}")
        drawn = group.sample(n=per_class, random_state=seed)
        train_parts.append(drawn.iloc[: n_train // 2])
        test_parts.append(drawn.iloc[n_train // 2 :])

    train = pd.concat(train_parts).sample(frac=1, random_state=seed)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)
