"""Print the numbers in every table of the report and the diagnosis numbers quoted in the text, from saved results.
Usage: python -m analysis.tables
"""
import json
import torch

NAMES = dict(adamw="AdamW", muon="Muon", newton_muon="Newton-Muon", snowball="Snowball", snowflake="Snowflake")
pm = lambda m, s, d: f"{m:.{d}f} +- {s:.{d}f}"

print("== Main results table (mean +- std over 3 seeds) ==")
summ = json.load(open("results/final/summary.json"))
for ds, d in [("mnist", 4), ("cifar10", 3)]:
    runs = torch.load(f"results/final/{ds}.pt", weights_only=False)["runs"]
    print(f"-- {ds}")
    for m in NAMES:
        r = summ[ds][m]
        ms = sum(sum(h["step_time"]) / len(h["step_time"]) for h in runs[m]) / len(runs[m])
        print(f"  {NAMES[m]:12s} train {pm(r['full_train_loss']['mean'], r['full_train_loss']['std'], d)}  "
              f"val {pm(r['val_loss']['mean'], r['val_loss']['std'], d)}  "
              f"test acc {pm(100 * r['test_acc']['mean'], 100 * r['test_acc']['std'], 2)}  ms/step {ms:.2f}   cfg {r['cfg']}")

print("\n== Hyperparameter table: selected configuration (lr, wd, gamma) ==")
for ds in ["mnist", "cifar10"]:
    sw = json.load(open(f"results/sweeps/{ds}.json"))
    print(f"  {ds:8s} " + "   ".join(f"{NAMES[m]}: {sw[m]['best']}" for m in NAMES))

print("\n== Damping table: best validation loss (seed 0) for each gamma ==")
for m in ["snowball", "snowflake"]:
    for ds in ["mnist", "cifar10"]:
        by_gamma = {}
        for r in json.load(open(f"results/sweeps/{ds}.json"))[m]["sweep"]:
            g = r["cfg"]["gamma"]; by_gamma[g] = min(by_gamma.get(g, float("inf")), r["val_loss"])
        print(f"  {NAMES[m]:9s} {ds:8s} " + "  ".join(f"{g:g}: {v:.4f}" for g, v in sorted(by_gamma.items())))

print("\n== Oracle table: Snowflake with estimated / exact / no second moment ==")
for ds in ["mnist", "cifar10"]:
    t = json.load(open(f"results/diagnosis/snowflake_oracle/{ds}.json"))["table"]
    for k, lab in [("muon", "Muon"), ("est", "Snowflake, estimated M"), ("oracle", "Snowflake, exact M"),
                   ("identity", "Snowflake, M = I")]:
        print(f"  {ds:8s} {lab:24s} val {t[k]['val_loss_mean']:.4f} +- {t[k]['val_loss_std']:.4f}   "
              f"test acc {t[k]['test_acc_mean']:.2f} +- {t[k]['test_acc_std']:.2f}   cfg {t[k]['cfg']}")

print("\n== Numbers quoted in the Snowflake diagnosis ==")
h = torch.load("results/diagnosis/effective_scaling.pt", weights_only=False)
per = torch.tensor(h["ratio_per_layer"]).mean(0)
print(f"  Muon effective scaling / spectral norm: mean {sum(h['ratio']) / len(h['ratio']):.2f}, "
      f"per layer {[round(x, 2) for x in per.tolist()]}   (cfg {h['cfg']})")
for ds in ["mnist", "cifar10"]:
    r = json.load(open(f"results/diagnosis/snowflake_diag/{ds}.json"))
    L, err = r["layers"], r["log"]["relerr"]
    first = [e[0] for e in err]
    print(f"  Snowflake, {ds}: last-step spectral norm / budget " + ", ".join(f"{v['sigma1_over_eps']:.1f}" for v in L.values())
          + "; held-out RMS " + ", ".join(f"{v['rms_val']:.2f}" for v in L.values())
          + "; held-out max " + ", ".join(f"{v['max_val']:.2f}" for v in L.values())
          + "; estimate error at step 50 " + ", ".join(f"{100 * e:.0f}%" for e in first)
          + ", at the end " + ", ".join(f"{100 * v['relerr_final']:.0f}%" for v in L.values()))
