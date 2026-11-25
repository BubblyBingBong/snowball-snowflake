"""Snowflake at its selected configuration: (1) accuracy of the running second-moment estimate M_t, compared every
50 steps with the exact M over the full training set under the current weights; (2) at the last step, the effective
scaling ||dW x|| / ||x|| of held-out inputs and the spectral norm of the update, both in units of the budget eps.
Usage: python -m experiments.snowflake_diag --dataset mnist|cifar10  ->  results/diagnosis/snowflake_diag/{dataset}.json
"""
import argparse, json, os
import torch
from core import load, train

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
ds = ap.parse_args().dataset
cfg = json.load(open(f"results/sweeps/{ds}.json"))["snowflake"]["best"]
data = load(ds)
(Xtr, _), (Xva, _), _ = data
log, last = dict(step=[], relerr=[[], [], []]), {}


@torch.no_grad()
def inputs_over(model, X, chunk=2500):
    """Per-layer inputs for all rows of X under the current weights, processed in chunks."""
    outs = [[] for _ in model.layers]
    for i in range(0, len(X), chunk):
        model(X[i:i + chunk])
        for l, z in enumerate(model.inputs):
            outs[l].append(z)
    return [torch.cat(o) for o in outs]


@torch.no_grad()
def exact_second_moments(model, X, chunk=2500):
    acc = None
    for i in range(0, len(X), chunk):
        model(X[i:i + chunk])
        part = [((u := z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)).T @ u).double() for z in model.inputs]
        acc = part if acc is None else [a + b for a, b in zip(acc, part)]
    return [a / len(X) for a in acc]


@torch.no_grad()
def callback(step, total, model, opt):
    if not (step % 50 == 0 or step == total):
        return
    params = opt.param_groups[0]["params"]
    M_true = exact_second_moments(model, Xtr)
    log["step"].append(step)
    for l, p in enumerate(params):
        M_est = opt.state[p]["K"].double()
        log["relerr"][l].append(((M_est - M_true[l]).norm() / M_true[l].norm()).item())
    if step == total:
        Z_val = inputs_over(model, Xva)
        for l, p in enumerate(params):
            dW, eps = opt.state[p]["update"], opt.state[p]["eps"]
            s = (Z_val[l] @ dW.T).norm(dim=1) / Z_val[l].norm(dim=1).clamp_min(1e-12) / eps
            last[f"layer{l + 1}"] = dict(sigma1_over_eps=torch.linalg.matrix_norm(dW, 2).item() / eps,
                                         rms_val=s.pow(2).mean().sqrt().item(), max_val=s.max().item(),
                                         relerr_final=log["relerr"][l][-1])


train("snowflake", cfg, data, dict(mnist=5, cifar10=10)[ds], callback=callback, record=True)
os.makedirs("results/diagnosis/snowflake_diag", exist_ok=True)
json.dump(dict(cfg=cfg, layers=last, log=log), open(f"results/diagnosis/snowflake_diag/{ds}.json", "w"), indent=1)
print(json.dumps(last, indent=1))
