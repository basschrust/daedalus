from unittest.mock import MagicMock, patch

import pytest

from run.simulation_main import main


@pytest.fixture
def mock_wig20_setup():
    """Prepare a minimal WIG20 configuration without external dependencies."""
    settings = {
        "days": 3,
        "sim_time": 50,
        "lookback_sessions": 252,
    }

    with (
        patch("run.simulation_main.load_dotenv"),
        patch("run.simulation_main.yaml.safe_load", return_value=settings),
        patch("run.simulation_main.Path.open", MagicMock()),
        # This test verifies failure AFTER a cache miss, not freshness logic.
        # Mocking Path.open globally also breaks zoneinfo's TZif file loading.
        patch("run.simulation_main.PreparedDatasetCache.use_if_fresh", return_value=False),
    ):
        yield


@pytest.mark.parametrize(
    "exception",
    [
        ConnectionError("Failed to login: No such host is known"),
        TimeoutError("Failed to connect: connection timed out"),
    ],
)
def test_snowflake_connection_failure(
    mock_wig20_setup,
    exception,
):
    """Connection failures must prevent preparation and simulation."""

    with (
        patch(
            "run.simulation_main.connect",
            side_effect=exception,
        ) as mock_connect,
        patch("marketsim.scenarios.wig20.prepare") as mock_prepare,
        patch("run.simulation_main.run_simulation") as mock_simulation,
    ):
        with pytest.raises(
            RuntimeError,
            match="Cannot connect to Ikaros Snowflake",
        ):
            main(["--dataset", "wig20", "--prepare-only"])

    mock_connect.assert_called_once()
    mock_prepare.assert_not_called()
    mock_simulation.assert_not_called()
