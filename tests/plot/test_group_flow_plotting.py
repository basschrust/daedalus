import pandas as pd
import pytest

from marketsim.plot.group_flows import plot_group_flows


def sample_flows():
    return pd.DataFrame(
        [
            dict(
                day=0,
                time_tick=1,
                asset_id=10,
                agent_group="MM",
                cp_group="Noise",
                volume_buy=4,
                volume_sell=0,
                cash_buy=40,
                cash_sell=0,
            ),
            dict(
                day=0,
                time_tick=1,
                asset_id=10,
                agent_group="MM",
                cp_group="Noise",
                volume_buy=6,
                volume_sell=0,
                cash_buy=60,
                cash_sell=0,
            ),
            dict(
                day=1,
                time_tick=1,
                asset_id=10,
                agent_group="MM",
                cp_group="HBL",
                volume_buy=0,
                volume_sell=3,
                cash_buy=0,
                cash_sell=30,
            ),
            dict(
                day=1,
                time_tick=2,
                asset_id=11,
                agent_group="MM",
                cp_group="Noise",
                volume_buy=1,
                volume_sell=0,
                cash_buy=15,
                cash_sell=0,
            ),
            dict(
                day=0,
                time_tick=1,
                asset_id=10,
                agent_group="Noise",
                cp_group="MM",
                volume_buy=0,
                volume_sell=10,
                cash_buy=0,
                cash_sell=100,
            ),
        ]
    )


def test_one_figure_per_source_group(tmp_path):
    paths = plot_group_flows(sample_flows(), output_dir=tmp_path, labels_map={10: "ABC", 11: "DEF"})
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_empty_flows_do_not_create_files(tmp_path):
    assert plot_group_flows(pd.DataFrame(), output_dir=tmp_path) == []
    assert not list(tmp_path.glob("*.png"))


def test_missing_required_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing group-flow columns"):
        plot_group_flows(pd.DataFrame([{"day": 0}]), output_dir=tmp_path)
