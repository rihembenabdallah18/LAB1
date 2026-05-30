"""Back-to-back paired bars: accuracy vs each ReCEval metric.

Reads outputs/eval_results/{accuracy.csv, receval_summary.csv} and writes
outputs/plots/accuracy_vs_receval.png.

Three panels share the same condition rows (sorted by accuracy, best on top).
In each panel the accuracy bar grows LEFT and the ReCEval metric grows RIGHT.
Bar lengths are min-max normalised per column so the two sides are visually
comparable; the real value is printed on every bar. A condition with a long
left bar but a short right bar (or vice versa) = accuracy and reasoning quality
disagree (the H3 point in the report).

Standalone mirror of the notebook cell so the figure can be regenerated
without re-running the whole notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "outputs" / "eval_results"
PLOTS = REPO_ROOT / "outputs" / "plots"

ALL_ORDER = [
    "baseline", "student_direct_ft", "student_set_a", "student_set_b",
    "student_set_c", "student_set_c_mix", "student_direct_ft_large",
    "student_set_a_large", "student_set_b_large", "student_set_c_large",
]
SHORT = {
    "baseline": "baseline", "student_direct_ft": "direct_ft",
    "student_set_a": "set_a", "student_set_b": "set_b",
    "student_set_c": "set_c", "student_set_c_mix": "set_c_mix",
    "student_direct_ft_large": "direct_ft (L)", "student_set_a_large": "set_a (L)",
    "student_set_b_large": "set_b (L)", "student_set_c_large": "set_c (L)",
}

ACC_COLOR = "#4c72b0"          # accuracy (left side, every panel)
REC_COLORS = {                  # one colour per ReCEval metric (right side)
    "intra_mean": "#dd8452",
    "inter_mean": "#55a868",
    "info_mean": "#c44e52",
}
PANELS = [
    ("intra_mean", "Intra-step coherence", "{:.3f}"),
    ("inter_mean", "Inter-step coherence", "{:.3f}"),
    ("info_mean", "Informativeness", "{:+.2f}"),
]


def _norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return s * 0 + 0.5
    return (s - lo) / (hi - lo)


def main() -> None:
    acc = pd.read_csv(EVAL / "accuracy.csv")
    acc["accuracy_pct"] = acc["accuracy"] * 100
    rec = pd.read_csv(EVAL / "receval_summary.csv")

    df = acc[["condition", "accuracy_pct"]].merge(
        rec[["condition", "intra_mean", "inter_mean", "info_mean"]], on="condition"
    )
    df = df[df["condition"].isin(ALL_ORDER)].copy()
    # best accuracy on top
    df = df.sort_values("accuracy_pct", ascending=True).reset_index(drop=True)
    df["label"] = df["condition"].map(SHORT)
    df["acc_norm"] = _norm(df["accuracy_pct"])

    y = range(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)

    for ax, (col, title, fmt) in zip(axes, PANELS):
        df[f"{col}_norm"] = _norm(df[col])
        # accuracy bars grow LEFT (negative), ReCEval grows RIGHT (positive)
        ax.barh(y, -df["acc_norm"], color=ACC_COLOR, height=0.7,
                edgecolor="white", linewidth=0.5)
        ax.barh(y, df[f"{col}_norm"], color=REC_COLORS[col], height=0.7,
                edgecolor="white", linewidth=0.5)

        for yi, (_, r) in zip(y, df.iterrows()):
            ax.text(-r["acc_norm"] - 0.03, yi, f"{r['accuracy_pct']:.1f}%",
                    ha="right", va="center", fontsize=7.5)
            ax.text(r[f"{col}_norm"] + 0.03, yi, fmt.format(r[col]),
                    ha="left", va="center", fontsize=7.5)

        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlim(-1.45, 1.45)
        ax.set_xticks([])
        ax.set_title(title, fontsize=11)
        ax.set_yticks(list(y))
        ax.set_yticklabels(df["label"], fontsize=9)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)

    axes[0].annotate("← accuracy", xy=(-0.7, len(df) - 0.3),
                     ha="center", fontsize=8.5, color=ACC_COLOR, fontweight="bold")
    fig.legend(
        handles=[mpatches.Patch(color=ACC_COLOR, label="Accuracy (left)")]
        + [mpatches.Patch(color=REC_COLORS[c], label=f"{t} (right)")
           for c, t, _ in PANELS],
        loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "accuracy_vs_receval.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
