import pandas as pd
import pytest

from marketsim.plot.execution_volumes import EXECUTION_VOLUME_SQL, plot_execution_volumes_by_asset


def test_sql_aggregates_by_group_side_and_mode():
    duckdb = pytest.importorskip('duckdb')
    conn = duckdb.connect(':memory:')
    conn.execute('CREATE TABLE agents(agent_id INTEGER, group_name VARCHAR)')
    conn.execute('CREATE TABLE trades(asset_id INTEGER, agent_id INTEGER, order_side INTEGER, executed_mode VARCHAR, executed_volume INTEGER)')
    conn.execute("INSERT INTO agents VALUES (1,'MM'),(2,'ZI'),(3,'MM')")
    conn.execute("INSERT INTO trades VALUES (5,1,1,'waited',10),(5,3,1,'waited',7),(5,1,1,'arrived',4),(5,2,-1,'arrived',21),(6,1,-1,'waited',2)")
    df = conn.execute(EXECUTION_VOLUME_SQL).fetchdf()
    row = df.query("asset_id == 5 and agent_group == 'MM' and order_side == 1 and executed_mode == 'waited'")
    assert len(row) == 1 and int(row.iloc[0].volume) == 17
    assert int(df['volume'].sum()) == 44


def test_plot_creates_one_png_per_asset(tmp_path):
    df = pd.DataFrame([
        (5, 'MM', 1, 'waited', 17), (5, 'MM', 1, 'arrived', 4),
        (5, 'ZI', -1, 'arrived', 21), (6, 'MM', -1, 'waited', 2),
    ], columns=['asset_id', 'agent_group', 'order_side', 'executed_mode', 'volume'])
    paths = plot_execution_volumes_by_asset(df, output_dir=tmp_path)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_empty_data_does_not_create_plots(tmp_path):
    assert plot_execution_volumes_by_asset(pd.DataFrame(), output_dir=tmp_path) == []
