import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytest

from marketsim.plot.group_flows import plot_agent_group_flows


def test_plot_two_files_for_multiple_days_and_groups(tmp_path):
    df = pd.DataFrame([
        dict(day=0, time_tick=2, asset_id=101, cp_group='Noise', volume_buy=10,
             volume_sell=0, cash_buy=100, cash_sell=0),
        dict(day=1, time_tick=2, asset_id=101, cp_group='HBL', volume_buy=0,
             volume_sell=4, cash_buy=0, cash_sell=45),
        dict(day=1, time_tick=3, asset_id=102, cp_group='Noise', volume_buy=2,
             volume_sell=0, cash_buy=30, cash_sell=0),
    ])
    paths = plot_agent_group_flows(df, agent_id=17, output_dir=tmp_path,
                                   labels_map={101: 'ABC', 102: 'XYZ'})
    assert {p.name for p in paths} == {'17_group_assets_flows.png',
                                       '17_group_cash_flows.png'}
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


def test_empty_data_does_not_create_files(tmp_path):
    assert plot_agent_group_flows(pd.DataFrame(), agent_id=17, output_dir=tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_missing_columns_raise(tmp_path):
    with pytest.raises(ValueError, match='Missing group-flow columns'):
        plot_agent_group_flows(pd.DataFrame([{'day': 0}]), agent_id=17,
                               output_dir=tmp_path)
