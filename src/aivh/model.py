"""Logistic regression on either feature set, plus the evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import features
from .preprocess import clean

FEATURE_SETS = {"baseline": features.baseline, "stylometric": features.stylometric}


def build(feature_set: str, C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("features", FEATURE_SETS[feature_set]()),
            ("clf", LogisticRegression(C=C, max_iter=5000, solver="liblinear")),
        ]
    )


def prepare(texts: pd.Series, do_clean: bool = True) -> list[str]:
    return [clean(t) for t in texts] if do_clean else list(texts)


def evaluate(pipe: Pipeline, X_test: list[str], y_test: np.ndarray) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    wrong = pred != y_test
    return {
        "accuracy": accuracy_score(y_test, pred),
        "cohen_kappa": cohen_kappa_score(y_test, pred),
        "f1_ai": f1_score(y_test, pred),
        "confusion": {"human_as_human": int(tn), "human_as_ai": int(fp),
                      "ai_as_human": int(fn), "ai_as_ai": int(tp)},
        # How confident the model is when it is wrong vs right. A calibrated
        # model's errors sit near 0.5; an overconfident one's sit near 0 or 1.
        "mean_confidence_correct": float(np.mean(np.abs(proba[~wrong] - 0.5)) + 0.5) if (~wrong).any() else None,
        "mean_confidence_wrong": float(np.mean(np.abs(proba[wrong] - 0.5)) + 0.5) if wrong.any() else None,
        "_proba": proba,
        "_pred": pred,
    }


def cv_accuracy(feature_set: str, X: list[str], y: np.ndarray, folds: int = 5, seed: int = 42) -> list[float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_score(build(feature_set), X, y, cv=cv, scoring="accuracy").tolist()


def top_features(pipe: Pipeline, k: int = 15) -> pd.DataFrame:
    """Largest positive (AI-leaning) and negative (human-leaning) coefficients."""
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": names, "coef": coef})
    ai = df.nlargest(k, "coef").assign(leans="ai")
    human = df.nsmallest(k, "coef").assign(leans="human")
    return pd.concat([ai, human]).reset_index(drop=True)


def error_table(test: pd.DataFrame, result: dict) -> pd.DataFrame:
    out = test.assign(p_ai=result["_proba"], pred=result["_pred"])
    out = out[out["pred"] != out["label"]].copy()
    out["distance_from_0_5"] = (out["p_ai"] - 0.5).abs()
    out["excerpt"] = out["text"].str.slice(0, 240).str.replace("\n", " ")
    return out.sort_values("distance_from_0_5")[["label", "pred", "p_ai", "excerpt"]]
