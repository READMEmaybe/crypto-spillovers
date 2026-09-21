from pathlib import Path
from src.spillovers.build_spillovers import _default_min_obs, _default_out, OUT


def test_default_min_obs_is_ninety_percent_of_window():
    assert _default_min_obs(200) == 180
    assert _default_min_obs(60) == 54


def test_default_out_path_tags_nonbaseline_window():
    assert _default_out(200) == OUT
    assert _default_out(60) == OUT.parent / "spillovers_w60.parquet"
