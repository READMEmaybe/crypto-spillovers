"""Spectral-radius (VAR-stability) robustness re-run of H1/H2/H3 + the mechanism.

Drops rolling windows whose fitted VAR is non-stationary (companion spectral radius
``rho >= threshold``; see ``src/spillovers/build_stability.py``) and re-estimates:

  * headline  -- the full Step-2 ladder (``run_all``) on the share DV;
  * mechanism -- the bivariate decomposition ``logA / logB / log_localvar``.

then compares against the frozen results (``step2_panel_coefs``, ``decomp_coefs``).

Every regression ``dropna``s its DV, so NaN-ing a window == dropping it, and this is
exactly apples-to-apples with the headline (``crypto_vol`` is global / pre-join and
never changes). Outputs:

  gold/step2_stability_coefs.parquet      headline coefs, tagged by threshold
  gold/decomp_stability_coefs.parquet     mechanism coefs, tagged by threshold
  gold/stability_comparison.parquet/.csv  filtered-vs-frozen deltas at the primary cut

Run:  uv run python -m src.models.robustness_stability                  # rho >= 1.0
      uv run python -m src.models.robustness_stability --thresholds 1.0,0.99
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.decomposition import _load_base, log_winsorize, run_measure
from src.models.panel_regression import run_all

GOLD = Path("data/parquet/gold")
ANALYSIS = GOLD / "step2_analysis.parquet"
STABILITY = GOLD / "spillovers_stability.parquet"
STEP2_COEFS = GOLD / "step2_panel_coefs.parquet"      # frozen headline
DECOMP_COEFS = GOLD / "decomp_coefs.parquet"           # frozen mechanism
STEP2_STAB_OUT = GOLD / "step2_stability_coefs.parquet"
DECOMP_STAB_OUT = GOLD / "decomp_stability_coefs.parquet"
CMP_OUT = GOLD / "stability_comparison.parquet"
CMP_CSV = GOLD / "stability_comparison.csv"

PRIMARY_THRESHOLD = 1.0

# Columns that come from each channel's bivariate VAR and so must be dropped on an
# unstable window. Headline = the row-normalized share; mechanism = the un-normalized
# decomposition pieces (where rho >= 1 actually blows up).
HEADLINE_COLS = {"equity": ["spill_equity"], "fx": ["spill_fx"]}
MECH_COLS = {
    "equity": ["A_equity", "B_equity", "localvar_equity"],
    "fx": ["A_fx", "B_fx", "localvar_fx"],
}
MECH_MEASURES = {
    "logA": ("A_equity", "A_fx"),
    "logB": ("B_equity", "B_fx"),
    "log_localvar": ("localvar_equity", "localvar_fx"),
}


def apply_filter(df: pd.DataFrame, stability: pd.DataFrame, threshold: float,
                 cols_by_channel: dict[str, list[str]]) -> pd.DataFrame:
    """Copy of ``df`` with the given VAR-derived columns NaN'd where ``sr_<channel> >= threshold``.

    Left-merges the per-window spectral radii on ``(date, country_id)`` -- preserving
    df's row order -- and masks each channel's columns. Windows with no radius (NaN)
    are left untouched (``NaN >= threshold`` is False). The sr helper columns are
    dropped so the result matches the input schema.
    """
    out = df.merge(stability[["date", "country_id", "sr_equity", "sr_fx"]],
                   on=["date", "country_id"], how="left")
    for chan, cols in cols_by_channel.items():
        mask = (out[f"sr_{chan}"] >= threshold).to_numpy()
        present = [c for c in cols if c in out.columns]
        out.loc[mask, present] = np.nan
    return out.drop(columns=["sr_equity", "sr_fx"])


def run_headline(analysis: pd.DataFrame, stability: pd.DataFrame,
                 threshold: float) -> pd.DataFrame:
    """Re-run the full ladder on the rho-filtered share DV; tag with threshold."""
    df = apply_filter(analysis, stability, threshold, HEADLINE_COLS)
    coefs = run_all(df)
    coefs["threshold"] = threshold
    return coefs


def run_mechanism(base: pd.DataFrame, stability: pd.DataFrame,
                  threshold: float) -> pd.DataFrame:
    """Re-run logA/logB/log_localvar on the rho-filtered components; tag with threshold."""
    fb = apply_filter(base, stability, threshold, MECH_COLS)
    frames = []
    for measure, (eq, fx) in MECH_MEASURES.items():
        coefs = run_measure(fb, eq, fx, log_winsorize, {"measure": measure})
        coefs["threshold"] = threshold
        frames.append(coefs)
    return pd.concat(frames, ignore_index=True)


def _compare_block(cur: pd.DataFrame, filt: pd.DataFrame, keys: list[str],
                   family: str) -> pd.DataFrame:
    m = cur.merge(filt, on=keys, suffixes=("_cur", "_flt"))
    m["family"] = family
    m["d_coef"] = m["coef_flt"] - m["coef_cur"]
    m["d_pval"] = m["pval_flt"] - m["pval_cur"]
    m["d_nobs"] = m["nobs_flt"] - m["nobs_cur"]
    m["sign_flip"] = np.sign(m["coef_cur"]) != np.sign(m["coef_flt"])
    m["sig_change"] = (m["pval_cur"] < 0.05) != (m["pval_flt"] < 0.05)
    cols = (["family"] + keys + ["coef_cur", "coef_flt", "d_coef",
            "pval_cur", "pval_flt", "d_pval", "nobs_cur", "nobs_flt", "d_nobs",
            "sign_flip", "sig_change"])
    return m[cols]


def compare(baseline_head: pd.DataFrame, baseline_mech: pd.DataFrame,
            filtered_head: pd.DataFrame, filtered_mech: pd.DataFrame) -> pd.DataFrame:
    """Tidy filtered-vs-baseline comparison (deltas, sign flips, significance crossings).

    The baseline is the unfiltered (rho=inf) re-run computed in the same pass, so the
    only thing that differs from the filtered run is the dropped windows -- the
    comparison isolates the filter's effect and cannot be contaminated by code/sample
    drift relative to a stored parquet.
    """
    head_cmp = _compare_block(baseline_head.drop(columns=["threshold"]),
                              filtered_head.drop(columns=["threshold"]),
                              ["channel", "spec", "term"], "headline")
    mech_cmp = _compare_block(baseline_mech.drop(columns=["threshold"]),
                              filtered_mech.drop(columns=["threshold"]),
                              ["measure", "channel", "spec", "term"], "mechanism")
    return pd.concat([head_cmp, mech_cmp], ignore_index=True)


def _check_baseline_matches_frozen(baseline_head: pd.DataFrame) -> None:
    """Confirm the rho=inf baseline reproduces the stored headline (so 'baseline' == 'current')."""
    if not STEP2_COEFS.exists():
        print("  (frozen step2_panel_coefs not found; skipping baseline cross-check)")
        return
    frozen = pd.read_parquet(STEP2_COEFS)
    m = frozen.merge(baseline_head.drop(columns=["threshold"]),
                     on=["channel", "spec", "term"], suffixes=("_frozen", "_base"))
    max_d = (m["coef_frozen"] - m["coef_base"]).abs().max()
    status = "OK" if max_d < 1e-6 else f"DIFFERS (max |dcoef|={max_d:.2e})"
    print(f"  baseline (rho=inf) vs frozen step2_panel_coefs: {status}")


def _print_summary(cmp: pd.DataFrame, thr: float) -> None:
    focal = cmp[
        ((cmp.family == "headline") & (cmp.term == "ci_x_cv")
         & cmp.spec.isin(["pooled_full", "twfe"]))
        | ((cmp.family == "headline") & (cmp.term == "errf_x_cv")
           & cmp.spec.isin(["fx_h3_direct", "fx_h3_direct_net"]))
        | ((cmp.family == "mechanism") & (cmp.term == "ci_x_cv") & (cmp.spec == "pooled_full"))
    ]
    cols = ["family", "measure", "channel", "spec", "term", "coef_cur", "coef_flt",
            "d_coef", "pval_cur", "pval_flt", "d_nobs", "sign_flip", "sig_change"]
    pd.set_option("display.width", 220)
    print(f"\n=== focal terms: frozen vs rho < {thr} ===")
    print(focal[cols].to_string(index=False))
    print(f"\nrows compared: {len(cmp)}  sign flips: {int(cmp.sign_flip.sum())}  "
          f"significance crossings: {int(cmp.sig_change.sum())}")


def main(thresholds: tuple[float, ...] = (PRIMARY_THRESHOLD,)) -> None:
    analysis = pd.read_parquet(ANALYSIS)
    stability = pd.read_parquet(STABILITY)
    base = _load_base()

    # rho=inf filters nothing (all radii are finite) -> the in-run unfiltered baseline.
    all_thr = (np.inf,) + tuple(thresholds)
    head = pd.concat([run_headline(analysis, stability, t) for t in all_thr],
                     ignore_index=True)
    mech = pd.concat([run_mechanism(base, stability, t) for t in all_thr],
                     ignore_index=True)
    head.to_parquet(STEP2_STAB_OUT, index=False)
    mech.to_parquet(DECOMP_STAB_OUT, index=False)

    base_head = head[head.threshold == np.inf]
    base_mech = mech[mech.threshold == np.inf]
    _check_baseline_matches_frozen(base_head)

    cmps = []
    for thr in thresholds:
        cmp = compare(base_head, base_mech,
                      head[head.threshold == thr], mech[mech.threshold == thr])
        cmp.insert(0, "threshold", thr)
        cmps.append(cmp)
    cmp_all = pd.concat(cmps, ignore_index=True)
    cmp_all.to_parquet(CMP_OUT, index=False)
    cmp_all.to_csv(CMP_CSV, index=False)

    prim = PRIMARY_THRESHOLD if PRIMARY_THRESHOLD in thresholds else thresholds[0]
    print(f"wrote {STEP2_STAB_OUT.name}, {DECOMP_STAB_OUT.name}, {CMP_OUT.name} "
          f"(thresholds={list(thresholds)})")
    _print_summary(cmp_all[cmp_all.threshold == prim], prim)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Spectral-radius robustness re-run")
    ap.add_argument("--thresholds", default="1.0",
                    help="comma-separated rho cutoffs (default 1.0)")
    args = ap.parse_args()
    main(tuple(float(x) for x in args.thresholds.split(",")))
