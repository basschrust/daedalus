import numpy as np
import pandas as pd


def safe_correlation(x, y) -> float:
    """Pearson correlation; NaN if undefined."""
    aligned = pd.concat(
        [pd.Series(x), pd.Series(y)],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return float("nan")

    a = aligned.iloc[:, 0].to_numpy(dtype=float)
    b = aligned.iloc[:, 1].to_numpy(dtype=float)

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])
