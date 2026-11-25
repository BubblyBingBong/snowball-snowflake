"""Final runs: every method at its selected configuration with seeds 0, 1, 2 (main results table, loss curves).
Methods whose selected configuration is unchanged reuse the runs already saved in results/final/{dataset}.pt.
Usage: python -m experiments.final_runs   ->  results/final/{mnist,cifar10}.pt, results/final/summary.json
"""
import json, os
import torch
from core import load, train, evaluate, make_snowball

SEEDS, EPOCHS = [0, 1, 2], dict(mnist=5, cifar10=10)
METHODS = ["adamw", "muon", "newton_muon", "snowball", "snowflake"]
os.makedirs("results/final", exist_ok=True)
summary = {}
for ds in ["mnist", "cifar10"]:
    data = load(ds)
    best = {m: v["best"] for m, v in json.load(open(f"results/sweeps/{ds}.json")).items()}
    path = f"results/final/{ds}.pt"
    old = torch.load(path, weights_only=False) if os.path.exists(path) else None
    runs, summary[ds] = {}, {}
    for m in METHODS:
        cfg = best[m]
        if old is not None and old["cfg"].get(m) == cfg:
            runs[m] = old["runs"][m]
            print(f"[{ds}] {m:12s} config unchanged, reusing saved runs", flush=True)
        else:
            runs[m] = []
            for s in SEEDS:
                kind = "snowflake" if m == "snowball" else m
                model, h = train(kind, cfg, data, EPOCHS[ds], seed=s, make_opt=make_snowball(cfg) if m == "snowball" else None)
                h["full_train_loss"], h["full_train_acc"] = evaluate(model, *data[0])
                h["test_loss"], h["test_acc"] = evaluate(model, *data[2])
                runs[m].append(h)
                print(f"[{ds}] {m:12s} seed {s}  val {h['val_loss'][-1]:.4f}  test acc {100 * h['test_acc']:.2f}%", flush=True)
        f = lambda key: torch.tensor([h[key] if not isinstance(h[key], list) else h[key][-1] for h in runs[m]])
        summary[ds][m] = dict(cfg=cfg, **{k: dict(mean=f(k).mean().item(), std=f(k).std().item())
                                          for k in ["full_train_loss", "val_loss", "test_loss", "test_acc"]})
    torch.save(dict(runs=runs, cfg=best), path)
json.dump(summary, open("results/final/summary.json", "w"), indent=1)
