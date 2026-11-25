"""Hyperparameter sweeps for every method (one seed, selection by final validation loss).

AdamW, Muon, Newton-Muon and Snowflake are swept over the grids below; Snowball is swept over learning rate and
damping with weight decay fixed to Muon's selected value (Snowball with gamma -> inf is Muon).
Usage: python -m experiments.sweep --dataset mnist|cifar10   ->  results/sweeps/{dataset}.json
"""
import argparse, itertools, json, math, os
import torch
from core import load, train, cfg_str, make_snowball

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
ds = ap.parse_args().dataset
EPOCHS = dict(mnist=5, cifar10=10)[ds]
LRS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
GRIDS = dict(
    adamw=dict(lr=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], wd=[0.0, 0.1, 1.0]),
    muon=dict(lr=LRS, wd=[0.0, 0.1, 0.3]),
    newton_muon=dict(lr=[0.002] + LRS, wd=[0.1, 0.3], gamma=[0.2, 1.0, 5.0]),
    snowflake=dict(lr=LRS, wd=[0.1, 0.3], gamma=[0.2, 1.0, 5.0, 20.0, 50.0, 100.0]),
    snowball=dict(lr=[0.005, 0.01, 0.02, 0.05, 0.1], wd=None, gamma=[0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 500.0]),
)
CACHE = f"results/sweeps/runs/{ds}"
os.makedirs(CACHE, exist_ok=True)
data, summary = load(ds), {}
for method, grid in GRIDS.items():
    if method == "snowball":
        grid = dict(grid, wd=[summary["muon"]["best"]["wd"]])
    rows = []
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(grid, vals))
        cache = f"{CACHE}/{method}__{cfg_str(cfg).replace(', ', '_')}.pt"
        if os.path.exists(cache):
            h = torch.load(cache)
        else:
            kind = "snowflake" if method == "snowball" else method
            _, h = train(kind, cfg, data, EPOCHS, make_opt=make_snowball(cfg) if method == "snowball" else None)
            torch.save(h, cache)
        vl = h["val_loss"][-1] if math.isfinite(h["val_loss"][-1]) else float("inf")
        rows.append(dict(cfg=cfg, val_loss=vl, val_acc=h["val_acc"][-1]))
        print(f"[{ds}] {method:12s} {cfg_str(cfg):32s} val_loss={vl:.4f}", flush=True)
    best = min(rows, key=lambda r: r["val_loss"])
    summary[method] = dict(best=best["cfg"], sweep=rows)
    print(f"[{ds}] BEST {method}: {cfg_str(best['cfg'])}  val_loss={best['val_loss']:.4f}", flush=True)
json.dump(summary, open(f"results/sweeps/{ds}.json", "w"), indent=1)
