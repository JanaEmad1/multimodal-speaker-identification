"""Draw the README results chart from the accuracies reported in the notebook.

    pip install matplotlib && python scripts/make_charts.py
"""
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})

# (setup, accuracy, test-set size, modality) -- from the notebook outputs / README table
RESULTS = [
    ("Audio only (MFCC)", 0.900, 10, "audio"),
    ("Early fusion (audio + text)", 0.900, 10, "fusion"),
    ("Late fusion (audio + text)", 0.900, 10, "fusion"),
    ("Text only (DistilBERT)", 0.200, 10, "text"),
    ("Text, DistilBERT + MLM fine-tuned", 0.269, 810, "text"),
]
COLOR = {"audio": BLUE, "fusion": AQUA, "text": ORANGE}


def wilson(p, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


fig, ax = plt.subplots(figsize=(8.5, 3.8))
for i, (name, acc, n, kind) in enumerate(reversed(RESULTS)):
    lo, hi = wilson(acc, n)
    ax.barh(i, acc, color=COLOR[kind], height=0.6)
    ax.errorbar(acc, i, xerr=[[acc - lo], [hi - acc]], color=INK, capsize=3, lw=1)
    ax.text(hi + 0.015, i, f"{acc:.0%}  (n={n})", va="center", color=INK, fontsize=9.5)
ax.axvline(0.2, color=GRAY, ls="--", lw=1.2)
ax.text(0.21, len(RESULTS) - 0.3, "chance = 20% (5 speakers)", color=MUTED, fontsize=9, va="center")
ax.set_ylim(-0.6, len(RESULTS) - 0.05)
ax.set_yticks(range(len(RESULTS)), [r[0] for r in reversed(RESULTS)])
ax.set_xlim(0, 1.2)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="y", visible=False)
ax.set(xlabel="Speaker-ID accuracy (whiskers: 95% confidence interval)",
       title="Speaker identity lives in the audio; text adds nothing")
fig.tight_layout()
fig.savefig(DOCS / "accuracy_by_modality.png", dpi=150)
for name, acc, n, _ in RESULTS:
    print(f"{name:36s} {acc:.3f}  95% CI {wilson(acc, n)[0]:.3f}-{wilson(acc, n)[1]:.3f}")
