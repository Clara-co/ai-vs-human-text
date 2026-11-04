"""Two feature sets: a word n-gram baseline and the stylometric model.

baseline     word unigrams + bigrams (what the first LightSIDE model used)
stylometric  five feature groups that describe *how* something is written:
    word     word unigrams (catches "because", "would", "n't")
    sub      character 3-5 grams inside word boundaries: prefixes and suffixes
             such as "-ally" and "-nce"
    punct    punctuation rhythm: every word replaced by "w", punctuation and
             line ends kept, then 2-4 grams ("w , w", "w . EOL")
    start    the first two tokens of each sentence ("^in conclusion", "^so ,"),
             which separates the discourse marker "So," from a mid-sentence "so"
    layout   eight numeric descriptors of sentence and line structure
"""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

MIN_DF = 2  # drop features seen in only one text; removes most one-off misspellings

_TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|n't|[.,;:!?()\-—\"']")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def punct_stream(text: str) -> list[str]:
    """Words -> 'w', punctuation kept, line ends -> 'EOL'."""
    out = []
    for line in text.splitlines():
        for tok in _TOKEN.findall(line):
            out.append("w" if tok[0].isalpha() else tok)
        out.append("EOL")
    return out


def punct_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> list[str]:
    s = punct_stream(text)
    grams = []
    for n in range(n_min, n_max + 1):
        grams += [" ".join(s[i : i + n]) for i in range(len(s) - n + 1)]
    # Runs of plain words carry no punctuation information
    return [g for g in grams if set(g.split()) != {"w"}]


def sentence_starts(text: str) -> list[str]:
    starts = []
    for sent in _SENT_SPLIT.split(text):
        toks = [t.lower() for t in _TOKEN.findall(sent)][:2]
        if toks:
            starts.append("^" + " ".join(toks))
    return starts


class Layout(BaseEstimator, TransformerMixin):
    """Numeric descriptors of structure, standardised on the training data."""

    names = [
        "n_lines",
        "mean_line_words",
        "share_lines_end_punct",
        "mean_sentence_words",
        "sd_sentence_words",
        "type_token_ratio",
        "comma_per_sentence",
        "share_long_words",
    ]

    def _describe(self, text: str) -> list[float]:
        lines = [ln for ln in text.splitlines() if ln.strip()] or [""]
        sents = [s for s in _SENT_SPLIT.split(text) if s.strip()] or [""]
        words = _WORD.findall(text)
        sent_len = [len(_WORD.findall(s)) for s in sents]
        return [
            len(lines),
            np.mean([len(_WORD.findall(ln)) for ln in lines]),
            np.mean([ln.rstrip()[-1:] in ".!?" for ln in lines]),
            np.mean(sent_len),
            np.std(sent_len),
            len({w.lower() for w in words}) / max(len(words), 1),
            text.count(",") / len(sents),
            np.mean([len(w) >= 8 for w in words]) if words else 0.0,
        ]

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(self._matrix(X))
        return self

    def transform(self, X):
        return sparse.csr_matrix(self.scaler_.transform(self._matrix(X)))

    def _matrix(self, X):
        return np.array([self._describe(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names, dtype=object)


def baseline() -> CountVectorizer:
    return CountVectorizer(ngram_range=(1, 2), min_df=MIN_DF, binary=True)


def stylometric() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", CountVectorizer(min_df=MIN_DF, binary=True, token_pattern=r"(?u)\b\w+(?:'\w+)?\b")),
            ("sub", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=MIN_DF, sublinear_tf=True)),
            ("punct", CountVectorizer(analyzer=punct_ngrams, min_df=MIN_DF, binary=True)),
            ("start", CountVectorizer(analyzer=sentence_starts, min_df=MIN_DF)),
            ("layout", Layout()),
        ]
    )
