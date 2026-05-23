"""Stage 5a: accuracy on the GSM8K or SVAMP test set.

For each condition with a Stage-4 generations JSONL, parse the predicted
final answer (#### priority, last-number fallback) and compare to gold
with tolerance ``abs(pred - gold) < 1e-6``.

Pass --dataset svamp to score SVAMP generations instead of GSM8K.

Outputs (GSM8K, default):
  - outputs/eval_results/accuracy.csv
  - outputs/plots/accuracy_bar.png

Outputs (SVAMP):
  - outputs/eval_results/svamp/svamp_accuracy.csv
  - outputs/plots/svamp_accuracy_bar.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.parse_answer import parse_answer
from src.utils.runcard import finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "outputs" / "generations"
EVAL_DIR = REPO_ROOT / "outputs" / "eval_results"
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"

GSM8K_CONDITIONS = [
    "baseline",
    "student_direct_ft",
    "student_set_a",
    "student_set_b",
    "student_set_c",
    "student_set_c_mix",
    "student_direct_ft_large",
    "student_set_a_large",
    "student_set_b_large",
    "student_set_c_large",
]

SVAMP_CONDITIONS = [
    "baseline",
    "student_direct_ft",
    "student_set_a",
    "student_set_b",
    "student_set_c",
    "svamp_student_direct_ft",
    "svamp_student_set_a",
    "svamp_student_set_b",
    "svamp_student_set_c",
]

DEFAULT_CONDITIONS = GSM8K_CONDITIONS  # backward-compat

TOL = 1e-6


def _equal(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < TOL


def _score_file(path: Path) -> dict:
    n = correct = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cot = row.get("generated_cot") or ""
            gold = row.get("gold_answer")
            if isinstance(gold, str):
                gold = parse_answer(gold)
            pred = parse_answer(cot)
            n += 1
            if _equal(pred, gold):
                correct += 1
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "n", "correct", "accuracy"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def _plot_bar(rows: list[dict], path: Path, title: str = "test accuracy by condition") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [r["condition"] for r in rows]
    acc = [r["accuracy"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(conds, acc, color="steelblue")
    ax.set_xticklabels(conds, rotation=25, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_table(rows: list[dict]) -> None:
    headers = ("condition", "n", "accuracy")
    col_w = [max(len(h), 12) for h in headers]
    col_w[0] = max(col_w[0], max(len(r["condition"]) for r in rows))
    col_w[1] = max(col_w[1], max(len(str(r["n"])) for r in rows))
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(col_w))
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for r in rows:
        print(fmt.format(r["condition"], r["n"], f"{r['accuracy']:.2%}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "svamp"], default="gsm8k",
                    help="Which dataset's generations to score (default: gsm8k)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Override conditions list (defaults to dataset-specific list)")
    ap.add_argument("--gen-dir", type=Path, default=None,
                    help="Override generations directory")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Override output CSV path")
    ap.add_argument("--plot", type=Path, default=None,
                    help="Override output plot path")
    args = ap.parse_args()

    # Resolve dataset-specific defaults
    if args.dataset == "svamp":
        default_conditions = SVAMP_CONDITIONS
        default_gen_dir    = GEN_DIR / "svamp"
        default_csv        = EVAL_DIR / "svamp" / "svamp_accuracy.csv"
        default_plot       = PLOTS_DIR / "svamp_accuracy_bar.png"
        run_name           = "svamp_accuracy"
        plot_title         = "SVAMP test accuracy by condition"
    else:
        default_conditions = GSM8K_CONDITIONS
        default_gen_dir    = GEN_DIR
        default_csv        = EVAL_DIR / "accuracy.csv"
        default_plot       = PLOTS_DIR / "accuracy_bar.png"
        run_name           = "accuracy"
        plot_title         = "GSM8K test accuracy by condition"

    conditions = args.conditions or default_conditions
    gen_dir    = args.gen_dir   or default_gen_dir
    out_csv    = args.out_csv   or default_csv
    plot_path  = args.plot      or default_plot

    card = start("05a", run_name, {
        "dataset": args.dataset,
        "conditions": conditions,
        "tolerance": TOL,
    })

    rows: list[dict] = []
    inputs: list[str] = []
    missing: list[str] = []
    for cond in conditions:
        path = gen_dir / f"{cond}.jsonl"
        if not path.exists():
            missing.append(cond)
            continue
        inputs.append(str(path))
        rows.append({"condition": cond, **_score_file(path)})

    if not rows:
        finish(card, status="failed",
               notes=f"no generations found in {gen_dir}; missing: {missing}")
        raise SystemExit(f"No generation files found in {gen_dir}. Run Stage 4 first.")

    _write_csv(rows, out_csv)
    _plot_bar(rows, plot_path, title=plot_title)
    _print_table(rows)

    finish(
        card,
        metrics={
            "dataset": args.dataset,
            "acc_per_condition": {r["condition"]: r["accuracy"] for r in rows},
            "n_conditions_scored": len(rows),
        },
        inputs=inputs,
        outputs=[str(out_csv), str(plot_path)],
        notes=f"missing: {missing}" if missing else "",
    )

    if missing:
        print(f"\n!! Note: scored {len(rows)} of {len(conditions)} conditions; "
              f"missing generations for: {missing}")


if __name__ == "__main__":
    main()
