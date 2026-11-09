from unittest.mock import MagicMock, patch

import pytest
from snowflake.connector.errors import DatabaseError

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
    ):
        yield


@pytest.mark.parametrize(
    "exception",
    [
        DatabaseError(
            msg="Failed to login: No such host is known",
            errno=250001,
            sqlstate="08001",
        ),
        DatabaseError(
            msg="Failed to connect: connection timed out",
            errno=250001,
            sqlstate="08001",
        ),
    ],
)
def test_snowflake_connection_failure(
    mock_wig20_setup,
    exception,
):
    """Connection failures must prevent preparation and simulation."""

    with (
        patch(
            "marketsim.scenarios.wig20.connect",
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
