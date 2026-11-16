import pandas as pd
import pytest

from marketsim.plot.execution_volumes import plot_execution_volumes_by_asset


def test_zero_execution_option_still_has_chart(tmp_path):
    paths = plot_execution_volumes_by_asset(
        pd.DataFrame(columns=["asset_id", "agent_group", "order_side", "executed_mode", "volume"]),
        output_dir=tmp_path,
        labels_map={42: "WIG20 CALL"},
        include_asset_ids=[42],
    )
    assert len(paths) == 1
    assert paths[0].is_file()
    assert "WIG20_CALL" in paths[0].name


def test_option_execution_modes_reject_other(tmp_path):
    rows = pd.DataFrame(
        [
            dict(
                asset_id=42,
                agent_group="OptionMM",
                order_side=1,
                executed_mode="other",
                volume=3,
            ),
            dict(
                asset_id=42,
                agent_group="OptionMM",
                order_side=-1,
                executed_mode="arrived",
                volume=2,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unexpected execution modes in execution_volumes",
    ):
        plot_execution_volumes_by_asset(
            rows,
            output_dir=tmp_path,
            labels_map={42: "WIG20 CALL"},
        )
