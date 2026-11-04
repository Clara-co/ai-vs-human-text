"""Unit tests plus a smoke test on a small synthetic corpus.

The synthetic texts only check that the pipeline runs and can learn an obvious
signal. They say nothing about accuracy on the real dataset.
"""

import random

import pandas as pd
import pytest

from aivh import __main__ as runner
from aivh import data, features
from aivh.preprocess import clean


def synthetic_corpus(n_per_class: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    topics = ["school uniforms", "electric cars", "remote work", "city parks", "phone bans"]
    human = [
        "i think {t} are fine because my friends say so. we were getting tired of it, honestly.",
        "my mom says {t} matter because people would care. i don't know, it's being debated.",
        "so yeah {t} are kind of a big deal because nobody's getting what they need.",
    ]
    ai = [
        "In conclusion, {t} offer substantial benefits.\nAdditionally, they promote sustainability.",
        "Furthermore, {t} represent a significant opportunity.\nUltimately, stakeholders benefit.",
        "Overall, {t} are essential.\nMoreover, they foster innovation and collaboration.",
    ]
    rows = []
    for label, templates in [(0, human), (1, ai)]:
        for i in range(n_per_class):
            t = rng.choice(topics)
            text = " ".join(rng.choice(templates).format(t=t) for _ in range(3)) + f" ({i})"
            rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def test_clean_removes_quotes_and_citations():
    text = 'He said "this is a quotation that runs quite long" today [3] (Smith et al., 2019).\n> block quote'
    out = clean(text)
    assert "quotation" not in out
    assert "[3]" not in out and "Smith" not in out
    assert "block quote" not in out
    assert "QUOTE" in out


def test_clean_keeps_short_quotes():
    assert '"no"' in clean('She said "no" twice.')


def test_punct_stream_marks_words_and_line_ends():
    assert features.punct_stream("Hi, there.\nOk") == ["w", ",", "w", ".", "EOL", "w", "EOL"]


def test_sentence_starts_separates_so_comma():
    assert features.sentence_starts("So, it works. so it works") == ["^so ,", "^so it"]


def test_balanced_split_is_balanced_and_disjoint():
    df = synthetic_corpus()
    train, test = data.balanced_split(df, n_train=100, n_test=60, seed=1)
    assert train["label"].value_counts().tolist() == [50, 50]
    assert test["label"].value_counts().tolist() == [30, 30]
    assert not set(train["text"]) & set(test["text"])


def test_balanced_split_errors_when_too_small():
    with pytest.raises(ValueError):
        data.balanced_split(synthetic_corpus(10), n_train=40, n_test=40)


def test_full_run_on_synthetic_corpus(tmp_path):
    metrics = runner.main(["--n-train", "120", "--n-test", "100", "--cv-folds", "3"],
                          df=synthetic_corpus(), out=tmp_path)
    assert metrics["models"]["stylometric"]["accuracy"] > 0.9
    for f in ["metrics.json", "RESULTS.md", "top_features.csv", "errors.csv",
              "figures/top_features.png", "figures/confidence.png"]:
        assert (tmp_path / f).exists()
