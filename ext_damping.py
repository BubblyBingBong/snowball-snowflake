"""Extend the damping grids whose selected value was at the edge (report, Table 5):
  Snowball  (input_aware.py, method iamuon): gamma {0.05, 0.2, 1, 5} -> + {20, 100, 500}   (gamma -> inf is Muon)
  Snowflake (optim_compare.py):              gamma {0.2, 1, 5, 20}   -> + {50, 100}        (gamma -> inf is M = I)
Same lr / wd grids and the same run caches as the original sweeps, so existing configs are not retrained.
Selection: lowest final validation loss, seed 0.  -> results/paper/extended_damping.json
Usage: .venv/bin/python ext_damping.py
"""
import itertools, json, math, os
import torch
from optim_compare import load, train, cfg_str
from input_aware import InputAware

EPOCHS = dict(mnist=5, cifar10=10)
LRS_M = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
out = {}


def run(cache, kind, cfg, data, epochs, make_opt=None):
    if os.path.exists(cache):
        return torch.load(cache)
    _, h = train(kind, cfg, data, epochs, make_opt=make_opt)
    os.makedirs(os.path.dirname(cache), exist_ok=True); torch.save(h, cache)
    return h


for ds in ["mnist", "cifar10"]:
    data = load(ds)
    muon_wd = json.load(open(f"results/optim_compare/{ds}/summary.json"))["muon"]["best"]["wd"]
    grids = {
        "snowball": [dict(lr=lr, wd=muon_wd, gamma=g) for lr, g in
                     itertools.product([0.005, 0.01, 0.02, 0.05, 0.1], [0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 500.0])],
        "snowflake": [dict(lr=lr, wd=wd, gamma=g) for lr, wd, g in
                      itertools.product(LRS_M, [0.1, 0.3], [0.2, 1.0, 5.0, 20.0, 50.0, 100.0])],
    }
    out[ds] = {}
    for meth, cfgs in grids.items():
        rows = []
        for c in cfgs:
            if meth == "snowball":
                cache = f"results/input_aware/{ds}/runs/iamuon__{cfg_str(c).replace(', ', '_')}.pt"
                h = run(cache, "snowflake", c, data, EPOCHS[ds],
                        make_opt=lambda model, c=c: InputAware(model, "iamuon", c["lr"], c["wd"], c["gamma"]))
            else:
                cache = f"results/optim_compare/{ds}/runs/snowflake__{cfg_str(c).replace(', ', '_')}.pt"
                h = run(cache, "snowflake", c, data, EPOCHS[ds])
            vl = h["val_loss"][-1] if math.isfinite(h["val_loss"][-1]) else float("inf")
            rows.append(dict(cfg=c, val_loss=vl, val_acc=h["val_acc"][-1]))
            print(f"[{ds}] {meth:9s} {cfg_str(c):32s} val_loss={vl:.4f}", flush=True)
        best = min(rows, key=lambda r: r["val_loss"])
        by_gamma = {}
        for r in rows:
            g = r["cfg"]["gamma"]; by_gamma[g] = min(by_gamma.get(g, float("inf")), r["val_loss"])
        out[ds][meth] = dict(best=best["cfg"], best_val_loss=best["val_loss"], best_val_loss_by_gamma=by_gamma, sweep=rows)
        print(f"[{ds}] BEST {meth}: {cfg_str(best['cfg'])}  val_loss {best['val_loss']:.4f}   by gamma: "
              + "  ".join(f"{g:g}:{v:.4f}" for g, v in sorted(by_gamma.items())), flush=True)
os.makedirs("results/paper", exist_ok=True)
json.dump(out, open("results/paper/extended_damping.json", "w"), indent=1)
