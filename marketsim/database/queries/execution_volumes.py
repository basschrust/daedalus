"""Execution-volume aggregation query used by both simulation backends."""

EXECUTION_VOLUME_SQL = """
SELECT
    t.asset_id,
    COALESCE(a.group_name, 'UNGROUPED') AS agent_group,
    t.order_side,
    CASE WHEN LOWER(COALESCE(t.executed_mode, '')) IN ('waited', 'arrived')
         THEN LOWER(t.executed_mode) ELSE 'other' END AS executed_mode,
    SUM(t.executed_volume) AS volume
FROM trades t
LEFT JOIN agents a ON a.agent_id = t.agent_id
WHERE t.order_side IN (1, -1)
GROUP BY t.asset_id, COALESCE(a.group_name, 'UNGROUPED'),
         t.order_side, executed_mode
ORDER BY t.asset_id, agent_group, t.order_side, executed_mode
"""
