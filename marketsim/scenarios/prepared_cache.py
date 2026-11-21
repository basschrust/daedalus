"""Compatibility exports for the DuckDB prepared-dataset cache.

Database access lives in marketsim.database.prepared_cache. This module
only preserves imports used by scenario orchestration and existing tests.
"""

from marketsim.database.prepared_cache import (
    PreparedDatasetCache,
    required_cutoff,
    selection_signature,
)

__all__ = ["PreparedDatasetCache", "required_cutoff", "selection_signature"]
