"""Prepared cohort inputs and training-fold-only linear estimation."""
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import linear as L
DATA_DIR=Path('data').resolve()


def load_data(cancer='luad'):
    path=DATA_DIR/f'{cancer}.npz'
    with np.load(path,allow_pickle=False) as f: d=dict(f)
    required={'x','clinical','prior','time','event','patients','genes','full_rna','broad_genes'}
    if not required<=d.keys(): raise ValueError(f'Missing fields: {required-d.keys()}')
    n=len(d['time']); g=len(d['genes'])
    assert d['x'].shape==(n,1,g) and d['clinical'].shape[0]==n
    assert d['full_rna'].shape==(n,len(d['broad_genes']))
    assert d['prior'].shape==(g,) and len(d['patients'])==n and len(d['event'])==n
    assert len(set(d['patients']))==n and np.isin(d['event'],[0,1]).all()
    assert np.isfinite(d['time']).all() and (d['time']>=0).all()
    assert (d['event'][d['time']==0]==0).all()
    if cancer!='luad': assert (d['time']>0).all()
    assert np.isfinite(d['prior']).all() and ((d['prior']>=.1)&(d['prior']<=1)).all()
    for name in ['x','clinical','full_rna']: assert not np.isinf(d[name]).any(),name
    d['gene_idx']=np.arange(len(d['broad_genes'])); d['cancer']=cancer
    for split in range(1,11): split_data(d,split)
    return d


def split_data(d,split):
    ids={p:d[f'{p}_{split:02}'] for p in ['train','val','test']}
    joined=np.concatenate(list(ids.values()))
    np.testing.assert_array_equal(np.sort(joined),np.arange(len(d['time'])))
    for value in ids.values(): assert value.dtype.kind in 'iu' and len(value)>0
    return d['x'],ids


def scale_fit(x, cancer='luad'):
    if cancer != 'luad':
        scaler = StandardScaler().fit(x)
        return scaler.mean_, scaler.scale_
    mu, sd = x.mean(0), x.std(0)
    return mu, np.where(sd > 1e-8, sd, np.inf)


def apply_scale(x, mu, sd, cancer):
    if cancer == 'luad':
        return (x - mu) / sd
    out = x.copy()
    out -= mu
    out /= sd
    return out


def linear_blocks(d, x, fit, held, eta):
    if d['cancer'] == 'luad':
        rf, rh, _ = L.prepare(x[fit], d['time'][fit], d['event'][fit], [x[i] for i in held])
    else:
        mu, sd = scale_fit(x[fit, 0], d['cancer'])
        rf = {'rna': apply_scale(x[fit, 0], mu, sd, d['cancer'])}
        rh = [{'rna': apply_scale(x[i, 0], mu, sd, d['cancer'])} for i in held]
    z = np.abs(L.uni_score(rf['rna'], d['time'][fit], d['event'][fit]))
    u = d['prior'] ** eta
    idx = np.argsort(-z / u, kind='stable')[:max(L.KR)]
    median = np.nanmedian(d['clinical'][fit], 0)
    if d['cancer'] != 'luad':
        median = np.nan_to_num(median, nan=0.)
    assert np.isfinite(median).all(), 'clinical column entirely missing in fitting rows'
    clinic = np.where(np.isnan(d['clinical']), median, d['clinical'])
    cm, cs = scale_fit(clinic[fit], d['cancer'])
    rm, rs = scale_fit(x[fit, 0], d['cancer'])
    state = dict(idx=idx, clinical_median=median, clinical_mean=cm, clinical_sd=cs,
                 rna_mean=rm, rna_sd=rs)
    cf = (clinic[fit] - cm) / cs
    ch = [(clinic[i] - cm) / cs for i in held]
    return np.column_stack((cf, rf['rna'][:, idx])), [np.column_stack((c, r['rna'][:, idx])) for c, r in zip(ch, rh)], u[idx], state


def candidates(d, x, fit, held):
    nc = d['clinical'].shape[1]
    for eta in (0, 1, 2):
        af, ah, u, state = linear_blocks(d, x, fit, held, eta)
        for k in L.KR:
            columns = np.r_[np.arange(nc), nc + np.arange(k)]
            xf, xh = af[:, columns], [a[:, columns] for a in ah]
            for w in (0., 1.):
                weights = np.r_[np.full(nc, w), u[:k] / u[:k].mean()]
                for a in L.AS:
                    for rho in L.COOP_RHO:
                        for lam, coef in zip(L.LAMS, L.path(L.Coop(xf, d['time'][fit], d['event'][fit], nc, rho), a, weights)):
                            yield (eta, k, w, a, rho, lam), coef, xf @ coef, [h @ coef for h in xh], state


def fit_linear(d, x, fit, held, cfg):
    eta, k, w, a, rho, lam = cfg
    af, ah, u, state = linear_blocks(d, x, fit, held, eta)
    nc = d['clinical'].shape[1]
    columns = np.r_[np.arange(nc), nc + np.arange(k)]
    af, ah = af[:, columns], [h[:, columns] for h in ah]
    weights = np.r_[np.full(nc, w), u[:k] / u[:k].mean()]
    coef = L.path(L.Coop(af, d['time'][fit], d['event'][fit], nc, rho), a, weights)[L.LAMS.index(lam)]
    state.update(coef=coef, k=k)
    return state, af @ coef, [h @ coef for h in ah]
