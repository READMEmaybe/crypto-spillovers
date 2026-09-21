import pandas as pd

from src.sample.crisis import (
    annual_fx_depreciation,
    flag_crisis_years,
    parse_inflation_panel,
)

PARAMS = {"inflation_pct_min": 25.0, "fx_depreciation_min": 0.30}


def test_annual_fx_depreciation_is_yoy_change_of_year_end_rate():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2019-06-01", "2019-12-31", "2020-06-01", "2020-12-31"]),
        "country_id": "AAA",
        "fx_rate_vs_usd": [90.0, 100.0, 110.0, 130.0],  # year-end 100 -> 130
    })
    out = annual_fx_depreciation(df).set_index("year")
    # 2020 depreciation = 130/100 - 1 = 0.30 (local-per-USD rose -> local weakened)
    assert abs(out.loc[2020, "fx_depreciation"] - 0.30) < 1e-9
    assert pd.isna(out.loc[2019, "fx_depreciation"])  # no prior year


def test_flag_crisis_years_triggers_on_either_criterion():
    inflation = pd.DataFrame({
        "country_id": ["AAA", "BBB", "CCC"], "year": [2020, 2020, 2020],
        "inflation_pct": [30.0, 5.0, 4.0],
    })
    depreciation = pd.DataFrame({
        "country_id": ["AAA", "BBB", "CCC"], "year": [2020, 2020, 2020],
        "fx_depreciation": [0.0, 0.35, 0.01],
    })
    out = flag_crisis_years(inflation, depreciation, PARAMS).set_index("country_id")
    assert out.loc["AAA", "crisis"] == 1 and "inflation" in out.loc["AAA", "reason"]
    assert out.loc["BBB", "crisis"] == 1 and "fx" in out.loc["BBB", "reason"]
    assert out.loc["CCC", "crisis"] == 0 and out.loc["CCC", "reason"] == ""


def test_parse_inflation_panel_keeps_per_year_values():
    payload = [{"page": 1}, [
        {"countryiso3code": "AAA", "date": "2020", "value": 30.0},
        {"countryiso3code": "AAA", "date": "2019", "value": 12.0},
        {"countryiso3code": "ZZZ", "date": "2020", "value": 99.0},  # not in keep
    ]]
    out = parse_inflation_panel(payload, keep={"AAA"})
    got = {(r.country_id, r.year): r.inflation_pct for r in out.itertuples()}
    assert got == {("AAA", 2020): 30.0, ("AAA", 2019): 12.0}
