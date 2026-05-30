"""Side-by-side training-set-size bars for GSM8K and SVAMP.

Reads outputs/runs/02_filter.json and 02_filter_svamp.json and writes
outputs/plots/training_set_sizes.png.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
PLOTS = ROOT / "outputs" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

gsm = json.loads((RUNS / "02_filter.json").read_text())["metrics"]
svamp = json.loads((RUNS / "02_filter_svamp.json").read_text())["metrics"]

LABELS = ["Direct FT", "Set A", "Set B", "Set C", "Set C+Mix"]
COLORS = ["#1f77b4", "#ffbb78", "#ff7f0e", "#2ca02c", "#98df8a"]

gsm_sizes = [
    gsm["direct_ft_size"],
    gsm["set_a_size"],
    gsm["set_b_size"],
    gsm["set_c_size"],
    gsm["set_c_size"] * 2,   # Set C + same-sized Direct-FT sample = 6,778 (full set, pre val split)
]
svamp_sizes = [
    svamp["direct_ft_size"],
    svamp["set_a_size"],
    svamp["set_b_size"],
    svamp["set_c_size"],
    None,
]

fig, (ax_g, ax_s) = plt.subplots(1, 2, figsize=(12, 3.4))

bars_g = ax_g.barh(LABELS, gsm_sizes, color=COLORS)
for bar, label, v in zip(bars_g, LABELS, gsm_sizes):
    txt = f"{v:,}"
    if label == "Set C":
        txt += f"  ({gsm['n_calculator_edited']:,} edits)"
    ax_g.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
              txt, va="center", fontsize=9)
ax_g.set_xlabel("Training examples")
ax_g.set_title("GSM8K")
ax_g.set_xlim(0, max(gsm_sizes) * 1.25)
ax_g.grid(axis="x", alpha=0.3)
ax_g.invert_yaxis()

svamp_plot = [v if v is not None else 0 for v in svamp_sizes]
bars_s = ax_s.barh(LABELS, svamp_plot, color=COLORS)
for bar, label, v in zip(bars_s, LABELS, svamp_sizes):
    if v is None:
        ax_s.text(20, bar.get_y() + bar.get_height() / 2,
                  "(no mix variant)", va="center", fontsize=9,
                  style="italic", color="gray")
    else:
        txt = f"{v:,}"
        if label == "Set C":
            txt += f"  ({svamp['n_calculator_edited']:,} edits)"
        ax_s.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                  txt, va="center", fontsize=9)
ax_s.set_xlabel("Training examples")
ax_s.set_title("SVAMP")
ax_s.set_xlim(0, max(v for v in svamp_sizes if v is not None) * 1.35)
ax_s.grid(axis="x", alpha=0.3)
ax_s.invert_yaxis()

fig.suptitle("Training set sizes by condition", fontsize=12)
fig.tight_layout()
out = PLOTS / "training_set_sizes.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
print(f"GSM8K  Set B keep rate: {gsm['set_b_keep_rate']:.1%}  | "
      f"calc edits: {gsm['n_calculator_edited']:,}/{gsm['set_b_size']:,} "
      f"({gsm['n_calculator_edited']/gsm['set_b_size']:.1%})")
print(f"SVAMP  Set B keep rate: {svamp['set_b_keep_rate']:.1%}  | "
      f"calc edits: {svamp['n_calculator_edited']:,}/{svamp['set_b_size']:,} "
      f"({svamp['n_calculator_edited']/svamp['set_b_size']:.1%})")
