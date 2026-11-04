#!/usr/bin/env bash
# Downloads AI_Human.csv from Kaggle into data/raw/.
# Needs a Kaggle account and an API token in ~/.kaggle/kaggle.json
# (Kaggle -> Settings -> API -> Create New Token).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle datasets download -d shanegerami/ai-vs-human-text -p data/raw --unzip
ls -lh data/raw
