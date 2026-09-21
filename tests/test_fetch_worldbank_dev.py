import pandas as pd
import pytest
import requests

from scripts.fetch_worldbank_dev import (
    build_country_dev,
    fetch_indicator,
    parse_indicator,
)


def _wb_payload(iso3_values):
    # World Bank v2 JSON: [metadata, [records]]
    records = []
    for iso3, year, value in iso3_values:
        records.append(
            {"countryiso3code": iso3, "date": str(year), "value": value}
        )
    return [{"page": 1, "pages": 1}, records]


def test_parse_indicator_means_over_years_per_country():
    payload = _wb_payload([("AUS", 2019, 100.0), ("AUS", 2020, 200.0),
                           ("BGD", 2019, 50.0), ("BGD", 2020, None)])
    out = parse_indicator(payload, keep={"AUS", "BGD"})
    assert out == {"AUS": 150.0, "BGD": 50.0}  # None ignored


def test_parse_indicator_filters_unknown_countries():
    payload = _wb_payload([("AUS", 2019, 100.0), ("FRA", 2019, 999.0)])
    out = parse_indicator(payload, keep={"AUS"})
    assert set(out) == {"AUS"}


def test_parse_indicator_handles_malformed_payload():
    assert parse_indicator([], keep={"AUS"}) == {}
    assert parse_indicator([{"page": 1}, None], keep={"AUS"}) == {}


class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code} Error")
            err.response = self
            raise err

    def json(self):
        return self._json


class _FakeSession:
    """Returns a queued sequence of responses, counting calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self._responses.pop(0)


def test_fetch_indicator_retries_transient_400(monkeypatch):
    # The World Bank country/all endpoint throttles with sporadic 400s; a retry
    # on the same fixed URL should recover rather than abort the whole pipeline.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    payload = [{"page": 1, "pages": 1}, []]
    sess = _FakeSession([_FakeResp(400), _FakeResp(200, payload)])
    out = fetch_indicator("NY.GDP.PCAP.CD", session=sess)
    assert out == payload
    assert sess.calls == 2  # retried once after the transient 400


def test_fetch_indicator_reraises_non_transient_404(monkeypatch):
    # A real client error (wrong indicator code) must NOT be retried.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    sess = _FakeSession([_FakeResp(404)])
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_indicator("NOPE", session=sess)
    assert sess.calls == 1


def test_fetch_indicator_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    sess = _FakeSession([_FakeResp(400), _FakeResp(400), _FakeResp(400)])
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_indicator("NY.GDP.PCAP.CD", session=sess, max_attempts=3)
    assert sess.calls == 3


def test_build_country_dev_shapes(monkeypatch):
    def fake(indicator, keep):
        return {"NY.GDP.PCAP.CD": {"AUS": 50000.0, "BGD": 2000.0},
                "FS.AST.PRVT.GD.ZS": {"AUS": 140.0, "BGD": 40.0}}[indicator]
    monkeypatch.setattr("scripts.fetch_worldbank_dev.parse_indicator_or_fetch", fake)
    df = build_country_dev(["AUS", "BGD"])
    assert list(df.columns) == ["country_id", "log_gdp_pc", "priv_credit_gdp"]
    assert len(df) == 2
    aus = df.set_index("country_id").loc["AUS"]
    assert abs(aus["log_gdp_pc"] - __import__("math").log(50000.0)) < 1e-9
    assert aus["priv_credit_gdp"] == 140.0
