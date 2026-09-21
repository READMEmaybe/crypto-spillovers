"""Unit tests for src/features/crypto_ban.py (derive + load_episodes)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.features import crypto_ban

# Synthetic episode set exercising every interval shape. AAA/BBB/CCC/DDD must be
# valid panel ids; we point load_episodes at a stub countries.yaml so the
# country-membership check passes for the fake ids.
SYNTHETIC = textwrap.dedent("""\
    country_id,ban_type,start_date,end_date,source,note
    AAA,absolute,2020-06-01,,SRC-A,open-ended absolute
    BBB,banking,2018-04-06,2020-03-04,SRC-B,closed banking (lift in-sample)
    CCC,payment,2017-01-01,,SRC-C,pre-2019 start still active in-window
    DDD,banking,2021-01-01,,SRC-D,banking-only friction
""")

STUB_COUNTRIES = textwrap.dedent("""\
    countries:
      - id: AAA
        name: Aaa
      - id: BBB
        name: Bbb
      - id: CCC
        name: Ccc
      - id: DDD
        name: Ddd
      - id: ZZZ
        name: Zzz
""")


@pytest.fixture
def episode_files(tmp_path: Path):
    csv = tmp_path / "episodes.csv"
    csv.write_text(SYNTHETIC)
    yml = tmp_path / "countries.yaml"
    yml.write_text(STUB_COUNTRIES)
    return csv, yml


@pytest.fixture
def episodes(episode_files):
    csv, yml = episode_files
    return crypto_ban.load_episodes(path=csv, countries_path=yml)


def _idx(pairs):
    return pd.MultiIndex.from_tuples(
        [(c, pd.Timestamp(d)) for c, d in pairs], names=["country_id", "date"]
    )


def test_load_episodes_parses_and_marks_open_ended(episodes):
    assert len(episodes) == 4
    aaa = episodes[episodes["country_id"] == "AAA"].iloc[0]
    assert pd.isna(aaa["end_date"])  # open-ended -> NaT
    bbb = episodes[episodes["country_id"] == "BBB"].iloc[0]
    assert bbb["end_date"] == pd.Timestamp("2020-03-04")


def test_derive_open_ended_episode(episodes):
    idx = _idx([("AAA", "2020-05-31"), ("AAA", "2020-06-01"), ("AAA", "2025-12-31")])
    out = crypto_ban.derive(idx, episodes=episodes)
    assert list(out) == [0, 1, 1]  # off before start, on from start, still on


def test_derive_closed_episode_inside_vs_after(episodes):
    # BBB is banking -> excluded by default rule; include banking to test the dates.
    active = frozenset({"absolute", "payment", "banking"})
    idx = _idx([
        ("BBB", "2018-04-05"),  # before start
        ("BBB", "2019-06-01"),  # inside (between start and end)
        ("BBB", "2020-03-04"),  # exactly end (inclusive)
        ("BBB", "2020-03-05"),  # after end -> lifted
    ])
    out = crypto_ban.derive(idx, active_types=active, episodes=episodes)
    assert list(out) == [0, 1, 1, 0]


def test_derive_pre_2019_start_active_in_window(episodes):
    # CCC payment ban starts 2017 -> active throughout the 2019+ window.
    idx = _idx([("CCC", "2019-01-01"), ("CCC", "2022-07-15")])
    out = crypto_ban.derive(idx, episodes=episodes)
    assert list(out) == [1, 1]


def test_derive_banking_excluded_by_default_included_when_selected(episodes):
    idx = _idx([("DDD", "2021-06-01"), ("DDD", "2024-01-01")])
    # Default {absolute, payment}: banking-only DDD -> 0.
    default_out = crypto_ban.derive(idx, episodes=episodes)
    assert list(default_out) == [0, 0]
    # Widen to include banking -> 1.
    wide = crypto_ban.derive(
        idx, active_types=frozenset({"absolute", "payment", "banking"}), episodes=episodes
    )
    assert list(wide) == [1, 1]


def test_derive_country_with_no_episode_is_zero(episodes):
    idx = _idx([("ZZZ", "2019-01-01"), ("ZZZ", "2023-06-01")])
    out = crypto_ban.derive(idx, episodes=episodes)
    assert list(out) == [0, 0]
    assert out.dtype.kind == "i"


def test_derive_accepts_dataframe_columns(episodes):
    df = pd.DataFrame(
        {"country_id": ["AAA", "ZZZ"], "date": pd.to_datetime(["2021-01-01", "2021-01-01"])}
    )
    out = crypto_ban.derive(df, episodes=episodes)
    assert list(out) == [1, 0]


# ── validation ──────────────────────────────────────────────────────────────


def _write(tmp_path: Path, body: str):
    csv = tmp_path / "bad.csv"
    csv.write_text("country_id,ban_type,start_date,end_date,source,note\n" + body)
    yml = tmp_path / "countries.yaml"
    yml.write_text(STUB_COUNTRIES)
    return csv, yml


def test_load_episodes_rejects_bad_ban_type(tmp_path):
    csv, yml = _write(tmp_path, "AAA,outright,2020-01-01,,SRC,note\n")
    with pytest.raises(ValueError, match="invalid ban_type"):
        crypto_ban.load_episodes(path=csv, countries_path=yml)


def test_load_episodes_rejects_end_before_start(tmp_path):
    csv, yml = _write(tmp_path, "AAA,absolute,2020-06-01,2020-01-01,SRC,note\n")
    with pytest.raises(ValueError, match="end_date < start_date"):
        crypto_ban.load_episodes(path=csv, countries_path=yml)


def test_load_episodes_rejects_unknown_country(tmp_path):
    csv, yml = _write(tmp_path, "XXX,absolute,2020-06-01,,SRC,note\n")
    with pytest.raises(ValueError, match="not in panel"):
        crypto_ban.load_episodes(path=csv, countries_path=yml)


def test_load_episodes_rejects_empty_source(tmp_path):
    csv, yml = _write(tmp_path, "AAA,absolute,2020-06-01,,,note\n")
    with pytest.raises(ValueError, match="empty source"):
        crypto_ban.load_episodes(path=csv, countries_path=yml)


def test_load_episodes_warns_on_overlap(tmp_path):
    body = (
        "AAA,absolute,2020-01-01,2021-01-01,SRC,first\n"
        "AAA,absolute,2020-06-01,,SRC,overlapping second\n"
    )
    csv, yml = _write(tmp_path, body)
    with pytest.warns(UserWarning, match="overlapping"):
        crypto_ban.load_episodes(path=csv, countries_path=yml)


def test_real_episode_table_validates():
    eps = crypto_ban.load_episodes()
    assert len(eps) == 24
    assert set(eps["ban_type"]) <= crypto_ban.VALID_BAN_TYPES


def test_real_table_default_transitions():
    summary = crypto_ban.annual_summary()
    trans_countries = sorted({t.split(":")[0] for t in summary.attrs["transitions"]})
    assert trans_countries == ["CHN", "EGY", "KWT", "QAT", "RUS", "TUR"]
