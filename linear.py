import numpy as np
AS=[.1,.5,.9]
LAMS=[.5,.3,.2,.1,.05,.03]
KR=[20,50,200]
COOP_RHO=[.1,.3,1.]
KS=[20,50,100,200]
ARMS={"crd":(["rna","cis"],["rna","cis","res"],[0.,1.]),"rna":(["rna"],["rna"],[0.]),"res":(["res"],["res"],[0.]),"concat":(["rna","hm","cna"],["rna","hm","cna"],[0.])}
def fit_decomp(x):
    """Per-gene OLS RNA ~ HM450 + CNA on the fitting sample; returns parameters only."""
    r, d, c = x[:, 0], x[:, 1], x[:, 2]
    n = len(r); ridge = 1e-6 * n
    m = [v.mean(0) for v in (r, d, c)]
    rc, dc, cc = r - m[0], d - m[1], c - m[2]
    sdd, scc, sdc = (dc * dc).sum(0) + ridge, (cc * cc).sum(0) + ridge, (dc * cc).sum(0)
    srd, src = (rc * dc).sum(0), (rc * cc).sum(0)
    det = sdd * scc - sdc ** 2
    return {"m": m, "bd": (srd * scc - src * sdc) / det, "bc": (src * sdd - srd * sdc) / det}

def components(p, x):
    r, d, c = x[:, 0], x[:, 1], x[:, 2]
    cis = p["m"][0] + p["bd"] * (d - p["m"][1]) + p["bc"] * (c - p["m"][2])
    return {"rna": r, "hm": d, "cna": c, "cis": cis, "res": r - cis}

def standardizer(comps):
    return {k: (v.mean(0), v.std(0)) for k, v in comps.items()}

def standardize(comps, s):
    out = {}
    for k, v in comps.items():
        mu, sd = s[k]
        out[k] = np.where(sd > 1e-8, (v - mu) / np.where(sd > 1e-8, sd, 1.0), 0.0)
    return out

def uni_score(x, t, e):
    # univariate Cox score statistic at beta=0, vectorized over genes
    order = np.argsort(-t, kind="stable"); x, t, e = x[order], t[order], e[order]
    n = np.arange(1, len(t) + 1)[:, None]
    mean = np.cumsum(x, 0) / n; var = np.cumsum(x ** 2, 0) / n - mean ** 2
    return ((x - mean) * e[:, None]).sum(0) / np.sqrt((var * e[:, None]).sum(0) + 1e-12)

class CoxData:
    def __init__(self, X, t, e):
        o = np.argsort(-t, kind="stable")
        self.X, self.e, ts = X[o], e[o].astype(float), t[o]
        self.last = np.searchsorted(-ts, -ts, side="right") - 1
        self.first = np.searchsorted(-ts, -ts, side="left")
        self.n = len(t)

    def loss(self, b):
        r = self.X @ b; m = r.max(); w = np.exp(r - m)
        s0 = np.cumsum(w)[self.last]
        return float((self.e * (np.log(s0) + m - r)).sum() / self.n)

    def loss_grad(self, b):
        r = self.X @ b; m = r.max(); w = np.exp(r - m)
        s0 = np.cumsum(w)[self.last]
        c = self.e / s0
        a = np.cumsum(c[::-1])[::-1][self.first]  # sum over i with t_i <= t_k of e_i / S0_i
        return float((self.e * (np.log(s0) + m - r)).sum() / self.n), self.X.T @ (w * a - self.e) / self.n

def ci(r, t, e):
    return float(CIndex(np.asarray(r, np.float64), t, e)) if np.ptp(r) > 0 else 0.5

def prepare(x_fit, t_fit, e_fit, x_others):
    p = fit_decomp(x_fit)
    cf = components(p, x_fit)
    s = standardizer(cf)
    fit = standardize(cf, s)
    others = [standardize(components(p, x), s) for x in x_others]
    z = {k: np.abs(uni_score(v, t_fit, e_fit)) for k, v in fit.items()}
    ranking = {arm: np.argsort(-np.max([z[c] for c in spec[1]], 0), kind="stable")[:max(KS)]
               for arm, spec in ARMS.items()}
    return fit, others, ranking
def solve(data, lam, a, w, b0, L=1.0, tol=1e-7, max_iter=5000):
    """FISTA with backtracking + restart for NLL + lam*sum w_g[a|b_g| + (1-a)/2 b_g^2]."""
    rw = lam * (1 - a) * w
    smooth = lambda b: data.loss(b) + 0.5 * (rw * b) @ b
    b, y, tk = b0.copy(), b0.copy(), 1.0
    L = max(L / 2, 1e-3)
    for _ in range(max_iter):
        fy, gy = data.loss_grad(y)
        fy += 0.5 * (rw * y) @ y; gy = gy + rw * y
        while True:
            v = y - gy / L
            z = np.sign(v) * np.maximum(np.abs(v) - lam * a * w / L, 0.0)
            d = z - y
            if smooth(z) <= fy + gy @ d + 0.5 * L * d @ d + 1e-12:
                break
            L *= 2
        step = np.linalg.norm(z - b)
        restart = (y - z) @ (z - b) > 0
        tk1 = 1.0 if restart else (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = z if restart else z + (tk - 1) / tk1 * (z - b)
        b, tk = z, tk1
        if step <= tol * max(1.0, np.linalg.norm(b)):
            break
    return b, L

def path(data, a, w):
    b, L, out = np.zeros(data.X.shape[1]), 1.0, []
    for lam in LAMS:
        b, L = solve(data, lam, a, w, b, L)
        out.append(b.copy())
    return out
class Coop:
    """Cox_OS(eta_1 + eta_2) + (rho / 2) (d / n) mean((eta_1 - eta_2)^2); the first nv columns are view 1."""
    def __init__(self, X, t, e, nv, rho):
        self.cox, self.X = CoxData(X, t, e), X
        self.D = np.hstack([X[:, :nv], -X[:, nv:]])
        self.c = rho * e.sum() / len(t) ** 2

    def loss(self, b):
        r = self.D @ b
        return self.cox.loss(b) + 0.5 * self.c * r @ r

    def loss_grad(self, b):
        (l, g), r = self.cox.loss_grad(b), self.D @ b
        return l + 0.5 * self.c * r @ r, g + self.c * (self.D.T @ r)

def std_fit(A):
    mu, sd = A.mean(0), A.std(0)
    return mu, np.where(sd > 1e-8, sd, np.inf)
def CIndex(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    concord = 0.
    total = 0.
    eav_count = 0

    for i in range(N_test):

        #print(f'i:{i}, theta[i]:{theta[i]}')
        if ystatus_test[i] == 1:

            for j in range(N_test):
                #print(f'\tj:{j}, theta[j]:{theta[j]}')
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                        #print(f'i, j: {i},{j}')
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5
                        eav_count = eav_count + 1
                        #print("相等的对数")
                        #print(eav_count)
    if total == 0:
        return 0
    #print("concord:", concord,  "total:", total)
    return concord / total
