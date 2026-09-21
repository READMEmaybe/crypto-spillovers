"""Figure 3.2, Analytical data workflow and quality controls.

A monochrome, APA-style diagram for Chapter 3. Portrait orientation so it fits
the main text block (not the appendix). Left column = the sequential workflow
spine; right column = the four quality controls named in the Note:
missing not imputed, schema validation halts the build, choices fixed in
configuration, and dependency-tracked rebuilds. No data dependency, pure diagram.

Run:  uv run python scripts/make_fig_3_2.py
Out:  thesis/figures/fig_3_2_workflow.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#000000"          # boxes, flow arrows, titles
BODY = "#1A1A1A"         # box body text
GATE_FILL = "#E9E9E9"    # quality-control gate stages
QC_FILL = "#F4F4F4"      # quality-control callouts
QC_EDGE = "#9A9A9A"      # callout edges + annotation connectors
FAINT = "#CFCFCF"        # column divider

# Left workflow column and right quality-control column.
LX, LW = 0.045, 0.525
QX, QW = 0.610, 0.380


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "figure.dpi": 120,
    })


def box(ax, top, title, body, *, gate=False, title_size=8.6, body_size=7.3,
        x=LX, w=LW, h=0.090):
    """Rounded stage box anchored by its TOP edge; returns (x, y, w, h)."""
    y = top - h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=GATE_FILL if gate else "white", edgecolor=INK,
        linewidth=1.1, transform=ax.transAxes, zorder=3))
    ax.text(x + w / 2, top - 0.019, title, transform=ax.transAxes,
            ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=INK, zorder=4)
    ax.plot([x + 0.014, x + w - 0.014], [top - 0.035, top - 0.035],
            transform=ax.transAxes, color="#8C8C8C", lw=0.6, zorder=4)
    ax.text(x + w / 2, y + (h - 0.035) / 2, body, transform=ax.transAxes,
            ha="center", va="center", fontsize=body_size, color=BODY,
            linespacing=1.32, zorder=4)
    return (x, y, w, h)


def flow(ax, a, b):
    """Solid down-arrow from bottom-centre of a to top-centre of b."""
    ax.add_patch(FancyArrowPatch(
        (a[0] + a[2] / 2, a[1]), (b[0] + b[2] / 2, b[1] + b[3]),
        transform=ax.transAxes, arrowstyle="-|>", mutation_scale=12,
        lw=1.2, color=INK, shrinkA=1.5, shrinkB=1.5, zorder=2))


def callout(ax, stage, text, *, h):
    """Italic quality-control note level with `stage`, tied by a dashed line."""
    yc = stage[1] + stage[3] / 2
    y = yc - h / 2
    ax.add_patch(FancyBboxPatch(
        (QX, y), QW, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=QC_FILL, edgecolor=QC_EDGE, linewidth=0.7,
        transform=ax.transAxes, zorder=3))
    ax.text(QX + 0.015, yc, text, transform=ax.transAxes, ha="left",
            va="center", fontsize=6.7, style="italic", color="#333333",
            linespacing=1.32, zorder=4)
    ax.add_patch(FancyArrowPatch(
        (stage[0] + stage[2], yc), (QX, yc), transform=ax.transAxes,
        arrowstyle="-", lw=0.7, color=QC_EDGE, linestyle=(0, (3, 2)),
        shrinkA=0, shrinkB=0, zorder=1))


def main() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column headers + faint divider.
    ax.text(LX + LW / 2, 0.980, "Analytical workflow", ha="center",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(QX + QW / 2, 0.980, "Quality controls", ha="center",
            fontsize=9, fontweight="bold", color=INK)
    ax.plot([0.583, 0.583], [0.150, 0.965], color=FAINT, lw=0.7, zorder=0)

    # --- Workflow spine (top -> bottom) ----------------------------------
    raw = box(ax, 0.958,
              "Raw source inputs",
              "Cryptocurrency (Binance, 1-min)\n"
              "Domestic equity & FX (daily close)\n"
              "Institutional, development & global series",
              h=0.100, body_size=7.1)
    clean = box(ax, 0.832,
                "Cleaning and harmonisation",
                "Common ISO3 country–date identifiers\n"
                "FX standardised to local currency per USD\n"
                "Invalid- and duplicate-price removal",
                h=0.112)
    schema = box(ax, 0.688,
                 "Schema validation",
                 "Valid identifiers · positive prices · unique country–date key",
                 gate=True, h=0.064, body_size=6.9)
    screen = box(ax, 0.596,
                 "Frozen data-quality sample screen",
                 "Coverage ≥ .80;  staleness ≤ .20 (equity) / ≤ .40 (FX)\n"
                 "→ 86 countries  (77 equity, 81 FX)",
                 gate=True, h=0.100, title_size=8.3)
    panel = box(ax, 0.468,
                "Analytical variables and merged panel",
                "Daily realised variance & 21-day crypto-stress measure\n"
                "Squared-return domestic-market volatility proxies\n"
                "Annual moderators aligned to the daily country–day panel",
                h=0.116, title_size=8.3)
    emp = box(ax, 0.324,
              "Empirical analysis (two stages)",
              "Step 1, rolling bivariate VAR → generalized FEVD\n"
              "→ Bitcoin-associated variance shares (equity, FX)\n"
              "Step 2, panel regression of shares on moderators & controls",
              h=0.128, body_size=7.1)

    for a, b in ((raw, clean), (clean, schema), (schema, screen),
                 (screen, panel), (panel, emp)):
        flow(ax, a, b)

    # --- Quality-control callouts (the four controls in the Note) --------
    callout(ax, clean,
            "Missing observations are not imputed;\n"
            "one-day ±0.5 log-return spikes that\n"
            "reverse within two days are removed\n"
            "as data-feed errors.", h=0.112)
    callout(ax, schema,
            "A failed check halts the build, so\n"
            "malformed data cannot propagate.", h=0.064)
    callout(ax, screen,
            "Thresholds are fixed in version-\n"
            "controlled configuration files and\n"
            "applied before results are examined.", h=0.100)

    # --- Footer: build-level provenance control --------------------------
    ax.add_patch(FancyBboxPatch(
        (LX, 0.066), 0.935, 0.072,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=QC_FILL, edgecolor=QC_EDGE, linewidth=0.8,
        transform=ax.transAxes, zorder=3))
    ax.text(LX + 0.935 / 2, 0.102,
            "Coordinated by a Make dependency graph: each output declares its inputs, data acquisition is\n"
            "separated from local rebuilds, and a locked environment with version-controlled configuration\n"
            "preserves the source-to-output lineage.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, color="#303030", linespacing=1.3, zorder=4)

    ax.text(LX, 0.028, "Shaded boxes are quality-control gates that can halt the build.",
            transform=ax.transAxes, ha="left", va="center", fontsize=6.6,
            style="italic", color="#555555")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_3_2_workflow.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved fig_3_2_workflow.pdf / .png -> {OUT}")


if __name__ == "__main__":
    main()
