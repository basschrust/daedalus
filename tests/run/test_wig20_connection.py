from unittest.mock import MagicMock, patch

import pytest
from snowflake.connector.errors import DatabaseError

from run.wig20 import main


@pytest.fixture
def mock_wig20_setup(tmp_path):
    """Prepare a minimal WIG20 configuration without external dependencies."""
    settings = {
        "days": 3,
        "sim_time": 50,
        "lookback_sessions": 252,
    }

    with (
        patch("run.wig20.load_dotenv"),
        patch("run.wig20.yaml.safe_load", return_value=settings),
        patch("run.wig20.Path.open", MagicMock()),
        patch("run.wig20.argparse.ArgumentParser.parse_args") as mock_args,
    ):
        mock_args.return_value = MagicMock(prepare_only=False)
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
    capsys,
    exception,
):
    """Connection failures should produce a friendly message, not a traceback."""

    with (
        patch("run.wig20.connect", side_effect=exception) as mock_connect,
        patch("run.wig20.prepare") as mock_prepare,
        patch("run.wig20.Simulator") as mock_simulator,
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 1

    output = capsys.readouterr()

    assert "ERROR: Cannot connect to Snowflake." in output.out
    assert "server may be unreachable" in output.out
    assert "Details:" in output.out

    assert output.err == ""

    mock_connect.assert_called_once()
    mock_prepare.assert_not_called()
    mock_simulator.assert_not_called()
