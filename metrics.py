"""The paper's pair-weighted horizon AUC, not an IPCW estimator."""
import numpy as np


def horizon_auc(risk, time, event, horizons=(12, 60, 120)):
    risk, time, event = [np.asarray(x, dtype=float).reshape(-1) for x in (risk, time, event)]
    if not (len(risk) == len(time) == len(event)) or not all(np.isfinite(x).all() for x in (risk, time, event)):
        raise ValueError('Finite, aligned risk, time and event vectors are required')
    if not np.isin(event, [0, 1]).all():
        raise ValueError('Event values must be 0 or 1')
    credit = (risk[:, None] > risk[None, :]) + .5 * (risk[:, None] == risk[None, :])
    rows = []
    for h in horizons:
        eligible = ((event == 1) & (time < h))[:, None] & (time > h)[None, :]
        den = int(eligible.sum())
        num = float(credit[eligible].sum())
        rows.append(dict(horizon=h, pairs=den, credit=num, auc=num / den if den else None))
    return rows


def pooled_auc(risk, time, event):
    rows = horizon_auc(risk, time, event)
    den = sum(r['pairs'] for r in rows)
    return sum(r['credit'] for r in rows) / den if den else None


def summary(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return dict(n_defined=len(values), mean=float(np.mean(values)) if values else None,
                sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
