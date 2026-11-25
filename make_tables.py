"""Print the numbers in every table of the report, and the diagnostic numbers quoted in the text, from saved results.
Usage: python make_tables.py
"""
import json
import torch

NAMES = dict(adamw="AdamW", muon="Muon", newton_muon="Newton-Muon", iam="Snowball", snowflake="Snowflake")


def pm(m, s, d):
    return f"{m:.{d}f} +- {s:.{d}f}"


print("== Main results table (Results section) (mean +- std over 3 seeds) ==")
summ = json.load(open("results/paper/summary.json"))
for ds, d in [("mnist", 4), ("cifar10", 3)]:
    runs = torch.load(f"results/paper/{ds}.pt", weights_only=False)["runs"]
    print(f"-- {ds}")
    for k in ["adamw", "muon", "newton_muon", "iam", "snowflake"]:
        r = summ[ds][k]
        ms = sum(sum(h["step_time"]) / len(h["step_time"]) for h in runs[k]) / len(runs[k])
        print(f"  {NAMES[k]:12s} train {pm(r['full_train_loss']['mean'], r['full_train_loss']['std'], d)}  "
              f"val {pm(r['val_loss']['mean'], r['val_loss']['std'], d)}  "
              f"test acc {pm(100 * r['test_acc']['mean'], 100 * r['test_acc']['std'], 2)}  ms/step {ms:.2f}   cfg {r['cfg']}")

print("\n== Oracle table (Appendix C): Snowflake with estimated / exact / no second moment ==")
for ds in ["mnist", "cifar10"]:
    t = json.load(open(f"results/snowflake_oracle/{ds}/summary.json"))["table"]
    for k, lab in [("muon", "Muon"), ("est", "Snowflake, estimated M"), ("oracle", "Snowflake, exact M"),
                   ("identity", "Snowflake, M = I")]:
        r = t[k]
        print(f"  {ds:8s} {lab:24s} val {r['val_loss_mean']:.4f} +- {r['val_loss_std']:.4f}   "
              f"test acc {r['test_acc_mean']:.2f} +- {r['test_acc_std']:.2f}   cfg {r['cfg']}")

print("\n== Hyperparameter table (appendix): selected values ==")
ext = json.load(open("results/paper/extended_damping.json"))
for ds in ["mnist", "cifar10"]:
    oc = json.load(open(f"results/optim_compare/{ds}/summary.json"))
    sel = {k: oc[k]["best"] for k in ["adamw", "muon", "newton_muon"]}
    sel["snowball"], sel["snowflake"] = ext[ds]["snowball"]["best"], ext[ds]["snowflake"]["best"]
    print(f"  {ds:8s} " + "   ".join(f"{k}: {v}" for k, v in sel.items()))

print("\n== Damping table (Appendix C): best validation loss (seed 0) for each damping gamma ==")
for meth in ["snowball", "snowflake"]:
    for ds in ["mnist", "cifar10"]:
        bg = ext[ds][meth]["best_val_loss_by_gamma"]
        print(f"  {meth:9s} {ds:8s} " + "  ".join(f"{float(g):g}: {v:.4f}" for g, v in sorted(bg.items(), key=lambda x: float(x[0]))))

print("\n== Numbers quoted in the Snowflake diagnosis (Discussion, Appendix C) ==")
h = torch.load("results/final_histories.pt", weights_only=False)
for k in ["muon", "adamw"]:
    r = torch.tensor(h[k]["ratio"]); per = torch.tensor(h[k]["ratio_per_layer"]).mean(0)
    print(f"  effective scaling / spectral norm, {k}: mean {r.mean():.2f}, per layer {[round(x, 2) for x in per.tolist()]}")
for ds in ["mnist", "cifar10"]:
    L = json.load(open(f"results/snowflake_diag/{ds}/summary.json"))["layers"]
    print(f"  Snowflake last step, {ds}: spectral norm / budget "
          + ", ".join(f"{v['sigma1_over_eps']:.1f}" for v in L.values())
          + "; held-out RMS " + ", ".join(f"{v['rms_val']:.2f}" for v in L.values())
          + "; held-out max " + ", ".join(f"{v['max_val']:.2f}" for v in L.values())
          + "; final estimate error " + ", ".join(f"{100 * v['relerr_final']:.0f}%" for v in L.values()))
