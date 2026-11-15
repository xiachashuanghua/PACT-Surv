"""Source-network primitives retained from the frozen implementation.

Use train.py --prepare-sources for target-specific exclusion and final settings.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
H1, H2, P_IN, P_HID = 256, 64, 0.2, 0.5
LR, WD, STEPS, EVERY, PATIENCE = 1e-5, .01, 5000, 1, 500
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class Strat:
    """Stratified Cox partial likelihood (Breslow ties), averaged over events: one block per cancer type, rows sorted by time
    descending, each block with its own max for a stable log-sum-exp. Types without events add nothing."""
    def __init__(self, t, e, g, dtype=torch.float32):
        self.blocks, self.ne = [], float(e.sum())
        for k in np.unique(g):
            idx = np.nonzero(g == k)[0]
            o = idx[np.argsort(-t[idx], kind="stable")]
            if e[o].sum() == 0:
                continue
            ts = t[o]
            key = np.r_[np.nonzero(ts[1:] != ts[:-1])[0], len(o) - 1]  # last row of each tie group
            last = np.repeat(key, np.diff(np.r_[-1, key]))
            self.blocks.append((torch.tensor(o, device=DEV), torch.tensor(last, device=DEV), torch.tensor(e[o], dtype=dtype, device=DEV)))

    def loss(self, r):
        """r: risks in the original row order."""
        tot = 0.0
        for o, last, ev in self.blocks:
            rb = r[o]
            m = rb.max().detach()
            s0 = torch.cumsum(torch.exp(rb - m), 0)[last]
            tot = tot + (ev * (torch.log(s0) + m - rb)).sum()
        return tot / self.ne


def np_strat_loss(r, t, e, g):
    """Reference: explicit loops."""
    tot = 0.0
    for i in np.nonzero(e == 1)[0]:
        rs = (g == g[i]) & (t >= t[i])
        tot += np.log(np.exp(r[rs]).sum()) - r[i]
    return tot / e.sum()


def strat_cindex(r, t, e, g):
    """Harrell C over comparable pairs within type (event i, t_j > t_i); ties in risk count 0.5."""
    num = den = 0.0
    for k in np.unique(g):
        m = g == k
        rk, tk, ek = r[m], t[m], e[m]
        i = np.nonzero(ek == 1)[0]
        comp = tk[None, :] > tk[i][:, None]
        d = rk[i][:, None] - rk[None, :]
        num += ((d > 0) * comp).sum() + 0.5 * ((d == 0) * comp).sum()
        den += comp.sum()
    return float(num / den)


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Dropout(P_IN), torch.nn.Linear(d, H1), torch.nn.GELU(), torch.nn.Dropout(P_HID),
                                     torch.nn.Linear(H1, H2), torch.nn.GELU(), torch.nn.Dropout(P_HID), torch.nn.Linear(H2, 1, bias=False))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def np_forward(W, X):
    """Dropout-free forward pass with numpy weights: (risk, last hidden layer)."""
    from scipy.special import erf
    gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
    h = gelu(X @ W["w1"].T + W["b1"])
    h2 = gelu(h @ W["w2"].T + W["b2"])
    return h2 @ W["w3"].ravel(), h2


def weights(net):
    s = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return {"w1": s["f.1.weight"], "b1": s["f.1.bias"], "w2": s["f.4.weight"], "b2": s["f.4.bias"], "w3": s["f.7.weight"]}


def train(X, t, e, g, tr, va, seed, steps=STEPS, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    Xt, Xv = torch.tensor(X[tr], device=DEV), torch.tensor(X[va], device=DEV)
    Lt, Lv = Strat(t[tr], e[tr], g[tr]), Strat(t[va], e[va], g[va])
    best, best_step, state, hist = np.inf, 0, None, []
    for step in range(1, steps + 1):
        net.train()
        opt.zero_grad()
        loss = Lt.loss(net(Xt))
        loss.backward()
        opt.step()
        if step % EVERY == 0:
            net.eval()
            with torch.no_grad():
                lv = float(Lv.loss(net(Xv)))
            hist.append((step, float(loss.detach()), lv))
            if lv < best - 1e-6:
                best, best_step, state = lv, step, {k: v.detach().clone() for k, v in net.state_dict().items()}
            elif step - best_step >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    return net, best, best_step, hist


def split(t, e, g):
    idx = np.arange(len(t))
    lab = np.char.add(g.astype(str), e.astype(int).astype(str))
    return train_test_split(idx, test_size=0.1, stratify=lab, random_state=19000)
