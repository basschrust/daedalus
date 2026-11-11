from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from marketsim.agent.agent import Agent
    from marketsim.market.instruments.security import Security

# Keep reference to the real console
terminal = sys.stdout

# remove printing to console
logger.remove()


# One file per each run:
def setup_logger(
    config, output_dir, subject: str = "market", subject_id: int | None = None
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if subject == "market":
        (Path(output_dir) / "market_logs").mkdir(parents=True, exist_ok=True)
    if subject == "agent":
        if subject_id is None:
            raise ValueError("subject id is required for agent logger")

        logger.add(
            sink=f"{output_dir}/agent_logs/agent_{subject_id}.log",
            format="{elapsed} | {message}",
            level="DEBUG" if config.debug_logging else "INFO",
            filter=lambda record, agent_id=subject_id: record["extra"].get("agent_id") == agent_id,
        )
    elif subject == "market":
        if subject_id is None:
            raise ValueError("subject id is required for market logger")
        logger.add(
            sink=Path(output_dir) / "market_logs" / f"market_{subject_id}.log",
            format="{elapsed} | {message}",
            level="DEBUG" if config.debug_logging else "INFO",
            filter=lambda record, market_id=subject_id: (
                record["extra"].get("market_id") == market_id
            ),
        )
    else:
        # so to the main
        logger.add(
            sink=output_dir / "main.log",
            format="{elapsed} | {message}",
            level="DEBUG" if config.debug_logging else "INFO",
        )


def setup_subject_logger(output_dir: Path = Path("tmp")) -> None:
    (Path(output_dir) / "market_logs").mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=Path(output_dir) / "market_logs" / "market_{extra[market_id]}.log",
        format="{elapsed} | {message}",
        level="DEBUG",
        filter=lambda record: "market_id" in record["extra"],
    )

    logger.add(
        sink=output_dir / "agent_{extra[agent_id]}.log",
        format="{elapsed} | {message}",
        level="DEBUG",
        filter=lambda record: "agent_id" in record["extra"],
    )


def setup_market_logger(*, market: Security, output_dir: Path | str = "tmp"):
    if output_dir == "tmp":
        return logger.bind(market_id=market.asset_id)
    output_dir = Path(output_dir)
    (output_dir / "market_logs").mkdir(parents=True, exist_ok=True)

    logger.add(
        output_dir / "market_logs" / f"market_{market.asset_id}_{market.short_name}.log",
        format="{elapsed} | {message}",
        level="DEBUG",
        filter=lambda record: record["extra"].get("market_id") == market.asset_id,
    )

    return logger.bind(market_id=market.asset_id)


def setup_agent_logger(*, agent: Agent, output_dir: Path | str = "tmp"):
    if output_dir == "tmp":
        return logger.bind(agent_id=agent.agent_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        output_dir / f"agent_logs/agent_{agent.agent_id}_{agent.group}.log",
        format="{elapsed} | {message}",
        level="DEBUG",
        filter=lambda record: record["extra"].get("agent_id") == agent.agent_id,
    )

    return logger.bind(agent_id=agent.agent_id)


class StreamToLogger:
    def write(self, log):
        log = log.strip()
        if log:
            logger.info(log)

    def flush(self):
        pass


sys.stdout = StreamToLogger()
# unhash if you want the errors be shown in the log, not console:
# sys.stderr = StreamToLogger()
