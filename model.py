"""Cox loss and standardized neural prediction head."""
import numpy as np
import torch

class CoxT:
    """The same partial likelihood in torch (float64), averaged over events."""
    def __init__(self, t, e):
        o = np.argsort(-t, kind="stable")
        ts = t[o]
        self.o = torch.as_tensor(o)
        self.last = torch.as_tensor(np.searchsorted(-ts, -ts, side="right") - 1)
        self.e = torch.as_tensor(e[o], dtype=torch.float64)
        self.ne = float(e.sum())

    def __call__(self, r):
        r = r[self.o]
        m = r.max().detach()
        s0 = torch.cumsum(torch.exp(r - m), 0)[self.last]
        return (self.e * (torch.log(s0) + m - r)).sum() / self.ne


class Head(torch.nn.Module):
    """Layers 2 and 3 of a pretrain19 network, initialized from its weights; output standardized as in nested19."""
    def __init__(self, W):
        super().__init__()
        self.l2 = torch.nn.Linear(W["w2"].shape[1], W["w2"].shape[0]).double()
        self.l3 = torch.nn.Linear(W["w3"].shape[1], 1, bias=False).double()
        with torch.no_grad():
            self.l2.weight.copy_(torch.as_tensor(W["w2"], dtype=torch.float64))
            self.l2.bias.copy_(torch.as_tensor(W["b2"], dtype=torch.float64))
            self.l3.weight.copy_(torch.as_tensor(W["w3"], dtype=torch.float64))
        self.mu, self.sd = float(W["risk_mean"]), float(W["risk_sd"])

    def forward(self, h):
        return (self.l3(torch.nn.functional.gelu(self.l2(h))).squeeze(-1) - self.mu) / self.sd
