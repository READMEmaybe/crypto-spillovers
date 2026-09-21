import numpy as np
import pandas as pd
from src.spillovers.build_decomposition import dxy_ret2, _wide_components


def test_dxy_ret2_is_squared_pct_change_by_date():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        "country_id": ["AUS", "AUS", "AUS"], "dxy": [100.0, 110.0, 99.0],
    })
    out = dxy_ret2(panel)
    # day2: pct change 0.10 -> 0.01 ; day3: -0.10 -> 0.01
    assert abs(out.set_index("date").loc[pd.Timestamp("2021-01-02"), "dxy_ret2"] - 0.01) < 1e-9
    assert abs(out.set_index("date").loc[pd.Timestamp("2021-01-03"), "dxy_ret2"] - 0.01) < 1e-9


def test_dxy_ret2_skips_gaps_without_losing_coverage():
    # DXY missing on day2; the return on day3 must be computed from the last
    # observed value (day1), not NaN-ed out by the gap.
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]),
        "country_id": ["AUS"] * 4, "dxy": [100.0, np.nan, 110.0, 99.0],
    })
    s = dxy_ret2(panel).set_index("date")["dxy_ret2"]
    assert pd.Timestamp("2021-01-02") not in s.index          # gap date dropped
    assert abs(s.loc[pd.Timestamp("2021-01-03")] - 0.01) < 1e-9  # (110/100-1)^2, across the gap
    assert abs(s.loc[pd.Timestamp("2021-01-04")] - 0.01) < 1e-9  # (99/110-1)^2


def test_wide_components_columns():
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    eq = pd.DataFrame({"spillover": [0.1]*3, "A": [0.5]*3, "B": [1.0]*3, "local_var": [2.0]*3}, index=idx)
    fx = pd.DataFrame({"spillover": [0.2]*3, "A": [0.3]*3, "B": [0.9]*3, "local_var": [1.5]*3}, index=idx)
    wide = _wide_components(eq, fx)
    assert set(wide.columns) == {"date", "A_equity", "B_equity", "localvar_equity",
                                 "A_fx", "B_fx", "localvar_fx"}
    assert len(wide) == 3
