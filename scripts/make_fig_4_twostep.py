"""Figure for the Chapter 4 (Econometric Methodology) intro, the two-stage design diagram.

A deliberately minimal, short banner showing the two-stage design: Step 1
(measurement) produces the Bitcoin-associated variance share, which becomes the
dependent variable in Step 2 (explanation). Monochrome, serif, ~6 text-lines tall
so it fits the small spot in the page.

Run:  uv run python scripts/make_fig_4_twostep.py
Out:  thesis/figures/fig_meth_twostep.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#000000"
BODY = "#1A1A1A"


def stage(ax, x, w, kicker, lines):
    """A single stage box: bold kicker, thin rule, body lines."""
    y, h = 0.12, 0.76
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.020",
        facecolor="white", edgecolor=INK, linewidth=1.2,
        transform=ax.transAxes, zorder=3))
    ax.text(x + w / 2, 0.745, kicker, transform=ax.transAxes, ha="center",
            va="center", fontsize=9.5, fontweight="bold", color=INK, zorder=4)
    ax.plot([x + 0.03, x + w - 0.03], [0.665, 0.665], transform=ax.transAxes,
            color="#8C8C8C", lw=0.6, zorder=4)
    ax.text(x + w / 2, 0.375, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=8.0, color=BODY,
            linespacing=1.45, zorder=4)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "figure.dpi": 120,
    })
    fig, ax = plt.subplots(figsize=(6.6, 1.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage(ax, 0.020, 0.420, "Step 1 · Measurement",
          ["Rolling bivariate VAR (Bitcoin RV + market)",
           "→ generalized FEVD →",
           "Bitcoin-associated variance share (equity, FX)"])
    stage(ax, 0.560, 0.420, "Step 2 · Explanation",
          ["Panel regression of the share on",
           "openness × crypto-stress (+ regime, controls)",
           "pooled OLS, Driscoll–Kraay standard errors"])

    ax.add_patch(FancyArrowPatch(
        (0.440, 0.50), (0.560, 0.50), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=14, lw=1.4, color=INK,
        shrinkA=0, shrinkB=0, zorder=2))
    ax.text(0.500, 0.705, "becomes the\ndependent\nvariable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=6.0, style="italic", color="#555555", linespacing=1.15)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_meth_twostep.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved fig_meth_twostep.pdf / .png -> {OUT}")


if __name__ == "__main__":
    main()
