"""Source-network fitting and fixed source-output standardization."""
import time
import numpy as np
import torch
import pretrain19 as B
STEPS, PATIENCE = 5000, 500

def labels(kind, s, t, e, g):
    """Real labels, or (time, event) pairs permuted within cancer type over all patients (as pretrain19.run)."""
    if kind == "real":
        return t, e
    rng = np.random.default_rng(100 + s)
    perm = np.arange(len(t))
    for k in np.unique(g):
        idx = np.nonzero(g == k)[0]
        perm[idx] = rng.permutation(idx)
    return t[perm], e[perm]


def fit(kind, s, lr, X, t, e, g, tr, va, steps=STEPS, patience=PATIENCE):
    B.LR = lr
    tl, el = labels(kind, s, t, e, g)
    t0 = time.time()
    net, best, step, hist = B.train(X, tl, el, g, tr, va, s, steps=steps, patience=patience)
    W = B.weights(net)
    r_tr, _ = B.np_forward(W, X[tr].astype(np.float64))
    r_va, _ = B.np_forward(W, X[va].astype(np.float64))
    with torch.no_grad():
        rt = net(torch.tensor(X[va], device=B.DEV)).cpu().numpy()
    fwd = float(np.abs(rt - r_va).max())
    assert fwd < 1e-3, fwd
    info = {"kind": kind, "seed": s, "lr": lr, "best_step": step, "best_val_loss": best, "steps_run": hist[-1][0],
            "seconds": round(time.time() - t0, 1), "val_cindex_real_labels": B.strat_cindex(r_va, t[va], e[va], g[va]),
            "train_cindex_real_labels": B.strat_cindex(r_tr, t[tr], e[tr], g[tr]), "max_abs_numpy_vs_torch": fwd,
            "risk_mean_train": float(r_tr.mean()), "risk_sd_train": float(r_tr.std())}
    del net
    torch.cuda.empty_cache()
    return {**W, "risk_mean": r_tr.mean(), "risk_sd": r_tr.std()}, info, hist
