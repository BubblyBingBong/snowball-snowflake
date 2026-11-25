"""Hypothesis 1 of the Snowflake diagnosis: do inputs lie along the top right singular vector of each update?
Trains Muon on MNIST at its selected configuration and records, at every step and for every layer, the batch mean of
||dW x|| / ||x|| divided by ||dW||_2 (1 would mean every input is stretched as much as the spectral norm allows).
At step 1000 it also records, for layer 2, the share of each input's squared norm along each right singular vector
of the update, (v_i . x)^2 / ||x||^2 (all mass on v_1 would make the effective scaling equal the spectral norm).
Usage: python -m experiments.effective_scaling   ->  results/diagnosis/effective_scaling.pt
"""
import json, os
import torch
from core import load, train, MatrixOpt

cfg = json.load(open("results/sweeps/mnist.json"))["muon"]["best"]
before, hist = [], dict(step=[], ratio=[], ratio_per_layer=[], cfg=cfg)


def make_opt(model):
    opt = MatrixOpt(model, "muon", cfg["lr"], cfg["wd"])
    opt.register_step_pre_hook(lambda o, a, k: before.__setitem__(slice(None), [l.weight.detach().clone() for l in model.layers]))
    return opt


@torch.no_grad()
def record(step, total, model, opt):
    per_layer = []
    for W0, layer, x in zip(before, model.layers, model.inputs):
        dW = layer.weight - W0
        eff = (x @ dW.T).norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)
        per_layer.append((eff / torch.linalg.matrix_norm(dW, ord=2)).mean().item())
    hist["step"].append(step); hist["ratio_per_layer"].append(per_layer); hist["ratio"].append(sum(per_layer) / len(per_layer))
    if step == 1000:                                              # singular-direction decomposition, layer 2
        dW, x = (model.layers[1].weight - before[1]).double(), model.inputs[1].double()
        _, _, Vh = torch.linalg.svd(dW, full_matrices=False)       # right singular vectors, sorted by singular value
        share = (x @ Vh.T) ** 2 / (x ** 2).sum(1, keepdim=True)    # (batch, d_in)
        hist["singular_share"] = dict(step=step, layer=2, mean=share.mean(0).cpu(),
                                      p10=share.quantile(0.1, dim=0).cpu(), p90=share.quantile(0.9, dim=0).cpu())


train("muon", cfg, load("mnist"), 5, make_opt=make_opt, callback=record)
os.makedirs("results/diagnosis", exist_ok=True)
torch.save(hist, "results/diagnosis/effective_scaling.pt")
print(f"mean ratio {sum(hist['ratio']) / len(hist['ratio']):.3f}")
