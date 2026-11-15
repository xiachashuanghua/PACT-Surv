"""Zero-reference Gauss-Legendre integrated gradients of the fitted transfer term."""
from types import SimpleNamespace
import numpy as np
import torch
import train as U
import model as M
DEVICE='cpu'
ARGS=SimpleNamespace(ig_nodes=32,ig_max_nodes=1024,patient_batch=8,
    quadrature_atol=1e-6,quadrature_rtol=1e-4,completeness_atol=1e-4,completeness_rtol=1e-3)

def finite(a, label):
    if not np.isfinite(a).all():
        raise ValueError(f"Non-finite values in {label}")


def require_close(actual, expected, label, atol, rtol):
    a, b = np.asarray(actual), np.asarray(expected)
    finite(a, label)
    finite(b, label)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shapes {a.shape} != {b.shape}")
    error = np.abs(a - b)
    if not np.all(error <= atol + rtol * np.abs(b)):
        raise AssertionError(f"{label}: max absolute discrepancy {float(error.max()):.9g}; "
                             f"required atol={atol}, rtol={rtol}")
    return float(error.max(initial=0.0))


def integrate_pre_activation(head, bias, delta, nodes, device, patient_batch):
    """Integrate the smooth continuation of the original fixed first-layer map."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    points = torch.as_tensor((points + 1.) / 2., dtype=torch.float64, device=device)
    weights = torch.as_tensor(weights / 2., dtype=torch.float64, device=device)
    bias = torch.as_tensor(bias, dtype=torch.float64, device=device)
    integrated = np.empty_like(delta, dtype=np.float64)
    for start in range(0, len(delta), patient_batch):
        end = min(start + patient_batch, len(delta))
        d = torch.as_tensor(delta[start:end], dtype=torch.float64, device=device)
        a = (bias + d[:, None, :] * points[None, :, None]).reshape(-1, d.shape[1])
        a.requires_grad_(True)
        output = head(torch.nn.functional.gelu(a, approximate="none"))
        gradient, = torch.autograd.grad(output.sum(), a)
        integral = (gradient.reshape(len(d), nodes, -1) * weights[None, :, None]).sum(1)
        integrated[start:end] = integral.detach().cpu().numpy()
    return integrated


def native_correction(z,nets,ft):
    if ft is None:return np.zeros(len(z)),np.zeros(len(z))
    values=[];baselines=[]
    for w,weights in zip(nets,ft['weights']):
        net=U.Network(w).eval()
        net.load_state_dict({k:torch.as_tensor(v) for k,v in weights.items()})
        with torch.no_grad():
            values.append(net(torch.as_tensor(z,dtype=torch.float32)).numpy())
            baselines.append(float(net(torch.zeros((1,z.shape[1]),dtype=torch.float32)).item()))
    return ft['gate']*np.stack(values).mean(0),np.full(len(z),ft['gate']*np.mean(baselines))


def neural_ig(z,nets,ft):
    if ft is None:return np.zeros_like(z,dtype=np.float64),np.zeros(len(z)),np.zeros(len(z)),[]
    native,reference=native_correction(z,nets,ft)
    attributed=np.zeros_like(z,dtype=np.float64);audits=[]
    for seed,(source,weights) in enumerate(zip(nets,ft['weights'])):
        # The fitted full-mode first layer, rather than its source initializer, defines the explanation.
        w=dict(source,w1=weights['l1.weight'],b1=weights['l1.bias'],w2=weights['l2.weight'],
               b2=weights['l2.bias'],w3=weights['l3.weight'])
        head=M.Head(w).to(DEVICE).eval()
        for p in head.parameters():p.requires_grad_(False)
        zz=torch.as_tensor(z,dtype=torch.float64,device=DEVICE)
        ww=torch.as_tensor(w['w1'],dtype=torch.float64,device=DEVICE)
        delta=(zz@ww.T).cpu().numpy()
        with torch.no_grad():
            b=torch.as_tensor(w['b1'],dtype=torch.float64,device=DEVICE)
            end=head(torch.nn.functional.gelu(torch.as_tensor(delta,device=DEVICE)+b)).cpu().numpy()
            base=float(head(torch.nn.functional.gelu(b[None,:])).item())
        change=end-base;nodes=ARGS.ig_nodes
        while True:
            integ=integrate_pre_activation(head,w['b1'],delta,nodes,DEVICE,ARGS.patient_batch)
            latent=(delta*integ).sum(1)
            if np.all(np.abs(latent-change)<=ARGS.quadrature_atol+ARGS.quadrature_rtol*np.abs(change)):break
            if nodes>=ARGS.ig_max_nodes:raise AssertionError(f'IG quadrature failed seed{seed}: {np.max(np.abs(latent-change))}')
            nodes=min(nodes*2,ARGS.ig_max_nodes)
        gradient=(torch.as_tensor(integ,device=DEVICE)@ww).cpu().numpy()
        scale=float(ft['gate'])/5
        ig=z.astype(np.float64)*gradient
        attributed+=scale*ig
        audits.append(dict(seed=seed,quadrature_nodes=nodes,
            quadrature_max_abs_error=float(np.max(np.abs(latent-change))),
            gene_projection_rounding_max_abs_error=float(np.max(np.abs(ig.sum(1)-latent))),
            scaled_gene_max_abs_completeness_error=float(np.max(np.abs(scale*(ig.sum(1)-change))))))
    # Original FP32 endpoints are retained. Their rounding residual is never assigned to genes.
    require_close(attributed.sum(1),native-reference,'unified IG native completeness',
                    ARGS.completeness_atol,ARGS.completeness_rtol)
    return attributed,native,reference,audits
