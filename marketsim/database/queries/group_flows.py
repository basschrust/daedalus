
# calculating group flows for L1

GROUP_FLOW_INSERT_SQL = '''
INSERT INTO l1_group_flows (
    day, time_tick, asset_id, agent_id, agent_group, cp_group,
    volume_buy, volume_sell, cash_buy, cash_sell
)
SELECT
    t.day, t.executed_time, t.asset_id, t.agent_id,
    COALESCE(a.group_name, 'UNGROUPED'),
    COALESCE(cp.group_name, 'UNGROUPED'),
    SUM(CASE WHEN t.order_side = 1 THEN t.executed_volume ELSE 0 END),
    SUM(CASE WHEN t.order_side = -1 THEN t.executed_volume ELSE 0 END),
    SUM(CASE WHEN t.order_side = 1 THEN ABS(t.cash) ELSE 0 END),
    SUM(CASE WHEN t.order_side = -1 THEN ABS(t.cash) ELSE 0 END)
FROM trades t
JOIN trades t_cp
    ON t_cp.day = t.day
   AND t_cp.asset_id = t.asset_id
   AND t_cp.order_id = t.matched_with
   AND t_cp.matched_with = t.order_id
   AND t_cp.executed_time = t.executed_time
   AND t_cp.order_side = -t.order_side
JOIN agents a ON a.agent_id = t.agent_id
JOIN agents cp ON cp.agent_id = t_cp.agent_id
GROUP BY t.day, t.executed_time, t.asset_id, t.agent_id,
         a.group_name, cp.group_name
'''
