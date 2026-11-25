"""Final 3-seed runs for the report: AdamW, Muon, Newton-Muon, IAM, Snowflake-RMS on the MLPs, at each method's
best config from its sweep (optim_compare.py for the first four, input_aware.py for IAM). Saves full histories so the
report figure can show mean and min-max across seeds.

Usage: .venv/bin/python paper_runs.py   ->  results/paper/{mnist,cifar10}.pt, results/paper/summary.json
"""
import json, os
import torch
from optim_compare import load, train, evaluate
from input_aware import InputAware

SEEDS = [0, 1, 2]
EPOCHS = dict(mnist=5, cifar10=10)
os.makedirs("results/paper", exist_ok=True)
summary = {}
for ds in ["mnist", "cifar10"]:
    data = load(ds)
    prev = json.load(open(f"results/optim_compare/{ds}/summary.json"))
    iam = json.load(open(f"results/input_aware/{ds}/summary.json"))["best"]["iamuon"]
    methods = {k: (k, prev[k]["best"], None) for k in ["adamw", "muon", "newton_muon", "snowflake"]}
    methods["iam"] = ("snowflake", iam, lambda model, c=iam: InputAware(model, "iamuon", c["lr"], c["wd"], c["gamma"]))
    out, summary[ds] = {}, {}
    for name, (kind, cfg, mk) in methods.items():
        out[name] = []
        for s in SEEDS:
            model, h = train(kind, cfg, data, EPOCHS[ds], seed=s, make_opt=mk)
            h["full_train_loss"], h["full_train_acc"] = evaluate(model, *data[0])
            h["test_loss"], h["test_acc"] = evaluate(model, *data[2])
            out[name].append(h)
            print(f"[{ds}] {name:12s} seed {s}  train {h['full_train_loss']:.4f}  val {h['val_loss'][-1]:.4f}  "
                  f"test acc {100*h['test_acc']:.2f}%", flush=True)
        f = lambda key: torch.tensor([h[key] if not isinstance(h[key], list) else h[key][-1] for h in out[name]])
        st = torch.tensor([torch.tensor(h["step_time"]).median().item() for h in out[name]])
        summary[ds][name] = dict(cfg=cfg, **{k: dict(mean=f(k).mean().item(), std=f(k).std().item()) for k in
                                 ["full_train_loss", "val_loss", "test_loss", "test_acc"]},
                                 median_step_ms=st.mean().item())
    torch.save(dict(runs=out, cfg={k: v[1] for k, v in methods.items()}), f"results/paper/{ds}.pt")
json.dump(summary, open("results/paper/summary.json", "w"), indent=1)
