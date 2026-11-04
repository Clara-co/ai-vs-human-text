"""Text cleaning that came out of the error analysis.

In the original LightSIDE project, quoted passages made essays look more
"human" than their author was: a quotation is someone else's writing. Block
quotes and long citations had the same effect. They are removed before any
features are built.
"""

from __future__ import annotations

import re

# A quotation of 6+ words in straight or curly double quotes
_LONG_QUOTE = re.compile(r"[\"“](?:[^\"“”]*?\s){5,}[^\"“”]*?[\"”]")
# Lines that start with ">" (block quotes)
_BLOCK_QUOTE = re.compile(r"^\s*>.*$", re.MULTILINE)
# Bracketed numeric citations: [1], [2, 3], [4-6]
_NUM_CITE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
# Parenthetical author-year citations: (Smith, 2019), (Smith et al., 2019; Lee 2020)
_AUTHOR_YEAR = re.compile(r"\((?:[A-Z][A-Za-z'\-]+(?: et al\.)?,? \d{4}[a-z]?(?:; )?)+\)")
_SPACES = re.compile(r"[ \t]+")


def clean(text: str) -> str:
    text = _BLOCK_QUOTE.sub(" ", text)
    text = _LONG_QUOTE.sub(" QUOTE ", text)
    text = _NUM_CITE.sub(" ", text)
    text = _AUTHOR_YEAR.sub(" ", text)
    text = _SPACES.sub(" ", text)
    # keep line breaks: line structure is itself a feature
    return "\n".join(line.strip() for line in text.splitlines()).strip()
