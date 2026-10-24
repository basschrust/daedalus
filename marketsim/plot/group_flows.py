"""Plot per-agent asset and cash transfers with counterparty groups."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


FLOW_COLUMNS = {'day', 'time_tick', 'asset_id', 'cp_group', 'volume_buy',
                'volume_sell', 'cash_buy', 'cash_sell'}


def plot_agent_group_flows(
    flows: pd.DataFrame,
    *,
    agent_id: int,
    output_dir: str | Path,
    labels_map: dict[int, str] | None = None,
) -> list[Path]:
    """Write two figures per agent (asset volume and cash), one panel per asset.

    Incoming assets (BUY) are positive; outgoing assets (SELL) are negative.
    Cash received on SELL is positive; cash paid on BUY is negative.
    Both figures distinguish counterparty groups and preserve day/tick order.
    """
    if flows.empty:
        return []
    missing = FLOW_COLUMNS - set(flows.columns)
    if missing:
        raise ValueError(f'Missing group-flow columns: {sorted(missing)}')

    df = flows.copy()
    df['cp_group'] = df['cp_group'].fillna('UNGROUPED').astype(str)
    df = df.sort_values(['day', 'time_tick', 'asset_id', 'cp_group'])
    # A compact integer x-coordinate for each observed (day, tick) pair avoids
    # guessing the number of ticks per day and handles sparse trading periods.
    moments = df[['day', 'time_tick']].drop_duplicates().reset_index(drop=True)
    moments['step'] = range(len(moments))
    df = df.merge(moments, on=['day', 'time_tick'], validate='many_to_one')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_map = labels_map or {}
    assets = sorted(df['asset_id'].unique())
    groups = sorted(df['cp_group'].unique())
    outputs = []

    for kind in ('assets', 'cash'):
        fig, axes = plt.subplots(len(assets), 1, figsize=(12, max(4, 3.5 * len(assets))),
                                 sharex=True, squeeze=False)
        for ax, asset_id in zip(axes[:, 0], assets):
            asset = df[df['asset_id'] == asset_id]
            for cp_group in groups:
                subset = asset[asset['cp_group'] == cp_group]
                if subset.empty:
                    continue
                if kind == 'assets':
                    subset = subset.assign(signed=subset['volume_buy'] - subset['volume_sell'])
                else:
                    subset = subset.assign(signed=subset['cash_sell'] - subset['cash_buy'])
                values = subset.groupby('step', as_index=False)['signed'].sum()
                ax.bar(values['step'], values['signed'], label=cp_group, alpha=0.75)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel('Shares (+in / -out)' if kind == 'assets' else 'Cash (+in / -out)')
            ax.set_title(labels_map.get(int(asset_id), str(asset_id)))
            ax.grid(axis='y', alpha=0.25)
            ax.legend(title='Counterparty group')
        # Keep labels readable for long simulations.
        stride = max(1, len(moments) // 12)
        ticks = moments.iloc[::stride]
        axes[-1, 0].set_xticks(ticks['step'])
        axes[-1, 0].set_xticklabels(
            [f'{int(day)}:{int(tick)}' for day, tick in zip(ticks['day'], ticks['time_tick'])],
            rotation=45, ha='right',
        )
        axes[-1, 0].set_xlabel('Day : time tick')
        fig.suptitle(f'Agent {agent_id} — {kind.capitalize()} flows by counterparty group')
        fig.tight_layout()
        path = output_dir / f'{agent_id}_group_{kind}_flows.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    return outputs
