"""Time-varying `crypto_ban` derived from a curated episode table.

Source of truth: `config/crypto_ban_episodes.csv`, one row per ban *episode*,
with a typed severity (`ban_type`). Countries with no row are legal throughout
(binary = 0). The panel binary is *derived* at daily granularity by a date-range
join, under a documented, toggleable rule over `ban_type`:

    crypto_ban[c, d] = 1  iff  ∃ episode e:
        e.country_id == c
        and e.ban_type ∈ ACTIVE_TYPES
        and e.start_date <= d <= (e.end_date or +∞)

Default `ACTIVE_TYPES = {absolute, payment}`, a *legal prohibition on use*.
`banking`-only friction is excluded by default but selectable for robustness.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path("config/countries.yaml")
EPISODES_CSV = Path("config/crypto_ban_episodes.csv")

VALID_BAN_TYPES = frozenset({"absolute", "payment", "banking"})
# Documented default rule: a legal prohibition on USE. banking-only bans are
# de-facto access friction (holding/trading legal) and are excluded by default.
DEFAULT_ACTIVE_TYPES = frozenset({"absolute", "payment"})

# Sample window for ban-day accounting / summaries.
SAMPLE_START = pd.Timestamp("2019-01-01")
SAMPLE_END = pd.Timestamp("2025-12-31")


def _panel_ids(path: Path = CONFIG) -> set[str]:
    """Country ids declared in countries.yaml (the authoritative panel universe)."""
    cfg = yaml.safe_load(Path(path).read_text())
    return {c["id"] for c in cfg["countries"]}


def load_episodes(
    path: Path = EPISODES_CSV,
    countries_path: Path = CONFIG,
) -> pd.DataFrame:
    """Read + validate the curated episode table.

    Returns a DataFrame with parsed `start_date`/`end_date` (NaT = open-ended).
    Raises ValueError on any schema violation; warns (does not raise) on
    overlapping episodes of the same country with the same ban_type.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    expected = ["country_id", "ban_type", "start_date", "end_date", "source", "note"]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path}: missing columns {missing_cols}")

    # ban_type enum
    bad_types = sorted(set(df["ban_type"]) - VALID_BAN_TYPES)
    if bad_types:
        raise ValueError(
            f"{path}: invalid ban_type(s) {bad_types}; allowed={sorted(VALID_BAN_TYPES)}"
        )

    # start_date: required, parseable
    start = pd.to_datetime(df["start_date"], format="%Y-%m-%d", errors="coerce")
    bad_start = df.loc[start.isna(), "start_date"].tolist()
    if bad_start:
        raise ValueError(f"{path}: unparseable start_date(s): {bad_start}")

    # end_date: empty (open-ended) or parseable; if present must be >= start_date
    end_raw = df["end_date"].str.strip()
    end = pd.to_datetime(end_raw.where(end_raw != ""), format="%Y-%m-%d", errors="coerce")
    bad_end_mask = (end_raw != "") & end.isna()
    if bad_end_mask.any():
        raise ValueError(
            f"{path}: unparseable end_date(s): {df.loc[bad_end_mask, 'end_date'].tolist()}"
        )
    reversed_mask = end.notna() & (end < start)
    if reversed_mask.any():
        offenders = df.loc[reversed_mask, ["country_id", "ban_type", "start_date", "end_date"]]
        raise ValueError(f"{path}: end_date < start_date for:\n{offenders.to_string(index=False)}")

    # country_id ∈ panel countries
    panel = _panel_ids(countries_path)
    unknown = sorted(set(df["country_id"]) - panel)
    if unknown:
        raise ValueError(f"{path}: country_id(s) not in panel countries.yaml: {unknown}")

    # source required
    empty_source = df["source"].str.strip() == ""
    if empty_source.any():
        offenders = df.loc[empty_source, ["country_id", "ban_type", "start_date"]]
        raise ValueError(f"{path}: empty source for:\n{offenders.to_string(index=False)}")

    out = df.copy()
    out["start_date"] = start
    out["end_date"] = end  # NaT = open-ended

    # WARN on overlapping episodes of the same (country_id, ban_type).
    for (cid, btype), grp in out.groupby(["country_id", "ban_type"]):
        grp = grp.sort_values("start_date")
        prev_end = None
        prev_start = None
        for _, row in grp.iterrows():
            if prev_start is not None:
                prev_stop = prev_end if pd.notna(prev_end) else pd.Timestamp.max
                if row["start_date"] <= prev_stop:
                    warnings.warn(
                        f"crypto_ban: overlapping {btype} episodes for {cid} "
                        f"(starts {prev_start.date()} and {row['start_date'].date()})",
                        stacklevel=2,
                    )
            prev_start, prev_end = row["start_date"], row["end_date"]

    return out.reset_index(drop=True)


def _extract_index(index) -> pd.DataFrame:
    """Normalize a (country_id, date) MultiIndex / DataFrame into a 2-col frame."""
    if isinstance(index, pd.DataFrame):
        if {"country_id", "date"}.issubset(index.columns):
            keys = index[["country_id", "date"]].copy()
        else:
            keys = index.index.to_frame(index=False)
    elif isinstance(index, pd.MultiIndex):
        keys = index.to_frame(index=False)
    else:
        raise TypeError(
            "derive() expects a (country_id, date) MultiIndex or a DataFrame "
            f"with those columns/index; got {type(index)!r}"
        )
    keys.columns = ["country_id", "date"]
    keys["date"] = pd.to_datetime(keys["date"])
    return keys


def derive(
    index,
    active_types=DEFAULT_ACTIVE_TYPES,
    episodes: pd.DataFrame | None = None,
) -> pd.Series:
    """Derive the daily 0/1 `crypto_ban` aligned to a (country_id, date) index.

    `index` may be a MultiIndex[country_id, date], or a DataFrame keyed by those
    (as columns or index). Returns an int Series (no NaN) aligned positionally to
    the input rows; countries with no qualifying episode get 0.
    """
    active = frozenset(active_types)
    if episodes is None:
        episodes = load_episodes()
    eps = episodes[episodes["ban_type"].isin(active)]

    keys = _extract_index(index)
    ban = pd.Series(0, index=keys.index, dtype="int64")

    if not eps.empty:
        by_country: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for _, e in eps.iterrows():
            end = e["end_date"] if pd.notna(e["end_date"]) else pd.Timestamp.max
            by_country.setdefault(e["country_id"], []).append((e["start_date"], end))

        for cid, intervals in by_country.items():
            cmask = keys["country_id"].to_numpy() == cid
            if not cmask.any():
                continue
            dates = keys.loc[cmask, "date"]
            active_here = pd.Series(False, index=dates.index)
            for start, end in intervals:
                active_here |= (dates >= start) & (dates <= end)
            ban.loc[active_here.index[active_here.to_numpy()]] = 1

    return ban.astype(int)


def annual_summary(
    active_types=DEFAULT_ACTIVE_TYPES,
    episodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-country ban-day accounting within the sample window + transition flags.

    Returns a DataFrame indexed by country_id with columns:
        ban_days: days with crypto_ban==1 in [2019-01-01, 2025-12-31]
        n_episodes: qualifying episodes (under active_types) for the country
        in_window_transition: True if the derived binary changes within the window
    Plus a `.attrs["transitions"]` list of human-readable transition strings.
    """
    active = frozenset(active_types)
    if episodes is None:
        episodes = load_episodes()
    eps = episodes[episodes["ban_type"].isin(active)]

    days = pd.date_range(SAMPLE_START, SAMPLE_END, freq="D")
    countries = sorted(set(episodes["country_id"]))

    rows = []
    transitions: list[str] = []
    for cid in countries:
        idx = pd.MultiIndex.from_product([[cid], days], names=["country_id", "date"])
        series = derive(idx, active_types=active, episodes=episodes)
        series.index = days
        ban_days = int(series.sum())
        n_eps = int((eps["country_id"] == cid).sum())

        diffs = series.diff().fillna(0)
        flip_dates = days[diffs.to_numpy() != 0]
        in_window_transition = len(flip_dates) > 0
        for fd in flip_dates:
            direction = "ON" if series.loc[fd] == 1 else "OFF"
            transitions.append(f"{cid}: {pd.Timestamp(fd).date()} -> ban {direction}")

        rows.append(
            {
                "country_id": cid,
                "ban_days": ban_days,
                "n_episodes": n_eps,
                "in_window_transition": in_window_transition,
            }
        )

    out = pd.DataFrame(rows).set_index("country_id")
    out.attrs["transitions"] = transitions
    return out
