"""Train and compare both models, write outputs/ and outputs/RESULTS.md.

    python -m aivh                      # 1,000 train / 1,000 test, as in the original study
    python -m aivh --n-train 20000      # same pipeline on a larger sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import data, model  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs"
BLUE, ORANGE, INK, INK_2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(argv=None, df=None, out: Path = OUT) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=data.DATA_PATH)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args(argv)

    (out / "figures").mkdir(parents=True, exist_ok=True)
    if df is None:
        df = data.load(args.data)
    train, test = data.balanced_split(df, args.n_train, args.n_test, args.seed)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()

    results, pipes = {}, {}
    for name, do_clean in [("baseline", False), ("stylometric", True)]:
        X_train = model.prepare(train["text"], do_clean)
        X_test = model.prepare(test["text"], do_clean)
        pipe = model.build(name).fit(X_train, y_train)
        r = model.evaluate(pipe, X_test, y_test)
        r["cv_accuracy"] = model.cv_accuracy(name, X_train, y_train, args.cv_folds, args.seed)
        r["n_features"] = len(pipe.named_steps["features"].get_feature_names_out())
        results[name], pipes[name] = r, pipe

    style = results["stylometric"]
    top = model.top_features(pipes["stylometric"])
    top.to_csv(out / "top_features.csv", index=False)
    errors = model.error_table(test, style)
    errors.to_csv(out / "errors.csv", index=False)

    plot_top_features(top, out / "figures" / "top_features.png")
    plot_confidence(test["label"].to_numpy(), style["_proba"], out / "figures" / "confidence.png")

    metrics = {
        "dataset_rows_after_dedup": len(df),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "models": {k: {m: v for m, v in r.items() if not m.startswith("_")} for k, r in results.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "RESULTS.md").write_text(write_results(metrics, top, errors))
    print(json.dumps(metrics["models"], indent=2))
    return metrics


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)


def plot_top_features(top, path, k: int = 10):
    df = top.groupby("leans", group_keys=False).head(k).sort_values("coef")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _style(ax)
    colors = [ORANGE if c > 0 else BLUE for c in df["coef"]]
    labels = [repr(f.split("__", 1)[1]) + f"  [{f.split('__', 1)[0]}]" for f in df["feature"]]
    ax.barh(range(len(df)), df["coef"], color=colors, height=0.65)
    ax.set_yticks(range(len(df)), labels, fontsize=8.5)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Logistic regression weight  (← human · AI →)", color=INK_2)
    ax.set_title("Strongest signals in the stylometric model", loc="left", fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confidence(y, proba, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    _style(ax)
    bins = np.linspace(0, 1, 21)
    ax.hist(proba[y == 0], bins=bins, color=BLUE, alpha=0.85, label="Human-written", edgecolor=SURFACE)
    ax.hist(proba[y == 1], bins=bins, color=ORANGE, alpha=0.85, label="AI-generated", edgecolor=SURFACE)
    ax.axvline(0.5, color=INK, linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted probability the text is AI-generated", color=INK_2)
    ax.set_ylabel("Test texts", color=INK_2)
    ax.set_title("How confident the model is on the test set", loc="left", fontweight="bold", color=INK)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_results(m: dict, top, errors) -> str:
    b, s = m["models"]["baseline"], m["models"]["stylometric"]

    def row(name, r):
        cv = r["cv_accuracy"]
        return (f"| {name} | {r['accuracy']:.3f} | {r['cohen_kappa']:.3f} | {r['f1_ai']:.3f} | "
                f"{np.mean(cv):.3f} ± {np.std(cv):.3f} | {r['n_features']:,} |")

    def feats(leans):
        rows = top[top["leans"] == leans].head(10)
        return ", ".join(f"`{f.split('__', 1)[1]}` ({f.split('__', 1)[0]})" for f in rows["feature"])

    c = s["confusion"]
    lines = [
        "# Results",
        "",
        "_Generated by `python -m aivh`. Do not edit by hand._",
        "",
        f"Corpus: {m['dataset_rows_after_dedup']:,} unique texts. Balanced sample: "
        f"{m['n_train']:,} train / {m['n_test']:,} test, seed {m['seed']}.",
        "",
        f"| Model | Test accuracy | Cohen's κ | F1 (AI) | {m['cv_folds']}-fold CV accuracy (train) | Features |",
        "|---|---|---|---|---|---|",
        row("Baseline: word 1-2 grams", b),
        row("Stylometric (cleaned text)", s),
        "",
        "## Stylometric model: confusion on the test set",
        "",
        "| | Predicted human | Predicted AI |",
        "|---|---|---|",
        f"| **Human** | {c['human_as_human']} | {c['human_as_ai']} |",
        f"| **AI** | {c['ai_as_human']} | {c['ai_as_ai']} |",
        "",
        "## What the model relies on",
        "",
        "![Top features](figures/top_features.png)",
        "",
        f"- **Human-leaning:** {feats('human')}",
        f"- **AI-leaning:** {feats('ai')}",
        "",
        "Full list: [`top_features.csv`](top_features.csv).",
        "",
        "## Confidence and errors",
        "",
        "![Confidence](figures/confidence.png)",
        "",
        f"Mean confidence when right: {s['mean_confidence_correct']:.2f}. "
        + (f"When wrong: {s['mean_confidence_wrong']:.2f}. " if s["mean_confidence_wrong"] else "")
        + f"{len(errors)} test texts were misclassified; [`errors.csv`](errors.csv) lists them, closest calls first.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
