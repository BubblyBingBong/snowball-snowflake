"""Snowflake diagnosis figure (appendix):
 (a) Muon on MNIST: batch mean of ||dW x||/||x|| divided by ||dW||_2 over training  (experiments/effective_scaling.py)
 (b) Muon on MNIST, layer 2, step 1000: share of ||x||^2 along each right singular vector of dW  (effective_scaling.py)
 (c) CIFAR-10 Snowflake: best validation loss vs damping gamma, estimated vs exact M  (sweep.py, snowflake_oracle.py)
 (d) CIFAR-10 test accuracy over 3 seeds: Snowflake estimated / exact / M = I vs Muon  (snowflake_oracle.py)
Usage: python -m plots.snowflake_diagnosis  ->  results/figures/snowflake_diagnosis.pdf"""
import os
import json
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

MUON, SF, SFX, NEUTRAL = "#eb6834", "#e87ba4", "#6250d6", "#8f8e86"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8})


def smooth(y, k=51):
    t = torch.tensor(y, dtype=torch.float64).view(1, 1, -1)
    return F.avg_pool1d(t, k, 1, padding=k // 2, count_include_pad=False).view(-1)


fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.8))
ax = ax.flatten()

# (a) hypothesis 1
h = torch.load("results/diagnosis/effective_scaling.pt", weights_only=False)
ax[0].plot(h["step"], smooth(h["ratio"]), color=MUON, lw=1.4)
ax[0].axhline(1.0, color=MUTED, lw=0.9, ls=":")
ax[0].annotate("inputs along top singular direction", (2150, 1.0), xytext=(0, -9), textcoords="offset points",
               ha="right", fontsize=7, color=MUTED)
ax[0].set_ylim(0, 1.1); ax[0].set_xlabel("step (MNIST)")
ax[0].set_ylabel("effective scaling / spectral norm")
ax[0].set_title("(a) Inputs see less than the spectral norm", loc="left", fontsize=8.5, color=INK)

# (b) hypothesis 1: where input energy lies among the update's singular directions
sh = h["singular_share"]
idx = torch.arange(1, len(sh["mean"]) + 1)
ax[1].fill_between(idx, sh["p10"], sh["p90"], color=MUON, alpha=0.18, lw=0, label="10th-90th percentile")
ax[1].plot(idx, sh["mean"], color=MUON, lw=1.0, label="batch mean")
ax[1].axhline(1 / len(idx), color=MUTED, lw=0.9, ls=":", label=f"evenly spread inputs (1/{len(idx)})")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel(f"right singular vector $v_i$ (layer {sh['layer']}, step {sh['step']})")
ax[1].set_ylabel(r"share of $\|x\|^2$ along $v_i$")
ax[1].set_title("(b) Inputs are not concentrated on $v_1$", loc="left", fontsize=8.5, color=INK)
ax[1].legend(frameon=False, fontsize=7, loc="lower center")
ax[0].annotate("Muon updates", (1100, 0.66), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK)

# (b) hypothesis 2: damping preference with estimated vs exact M
s = json.load(open("results/diagnosis/snowflake_oracle/cifar10.json"))
sweep = json.load(open("results/sweeps/cifar10.json"))["snowflake"]["sweep"]
for var, lab, col, mk, ls in [("est", "estimated $M$", SF, "o", (0, (4, 1.5))), ("oracle", "exact $M$", SFX, "s", "-")]:
    best = {}
    for r in (sweep if var == "est" else s["sweeps"][var]):
        g = r["cfg"]["gamma"]; best[g] = min(best.get(g, 9.0), r["val_loss"])
    gs = sorted(best)
    ax[2].plot(gs, [best[g] for g in gs], color=col, lw=1.4, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, label=lab)
GINF = 200.0                                   # plotted position of gamma = infinity (M = I)
ident = min(r["val_loss"] for r in s["sweeps"]["identity"])
ax[2].plot([20.0, GINF], [best[20.0], ident], color=SFX, lw=1.0, ls=":")
ax[2].plot([GINF], [ident], marker="D", ms=4.5, color=NEUTRAL, mec="white", mew=0.6, ls="none", label="$M = I$ ($\\gamma=\\infty$)")
mu = s["table"]["muon"]["val_loss_mean"]
ax[2].axhline(mu, color=MUON, lw=1.0, ls=":")
ax[2].annotate("Muon", (0.012, mu), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK)
ax[2].set_xscale("log"); ax[2].set_xlabel(r"damping $\gamma$ (CIFAR-10)"); ax[2].set_ylabel("best validation loss")
ax[2].xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, GINF]))
ax[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$\infty$" if v == GINF else f"{v:g}")); ax[2].xaxis.set_minor_formatter(NullFormatter())
ax[2].set_title("(c) Damping preference", loc="left", fontsize=8.5, color=INK)
ax[2].legend(frameon=False, fontsize=7, loc="upper right")

# (c) test accuracy, 3 seeds
order = [("est", "est. $M$", SF), ("oracle", "exact $M$", SFX),
         ("identity", "$M = I$", NEUTRAL), ("muon", "Muon", MUON)]
for i, (k, lab, col) in enumerate(order):
    accs = [100 * r["test_acc"] for r in s["final"][k]]
    m = sum(accs) / len(accs)
    ax[3].bar(i, m - 55, bottom=55, width=0.62, color=col, edgecolor="white", linewidth=1)
    ax[3].scatter([i] * len(accs), accs, s=7, color=INK, zorder=3, linewidths=0)
    ax[3].annotate(f"{m:.1f}", (i, max(accs)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK)
ax[3].set_xticks(range(len(order))); ax[3].set_xticklabels([o[1] for o in order], fontsize=7)
ax[3].set_xlabel("Snowflake variants vs. Muon")
ax[3].set_ylim(55, 60.6); ax[3].set_ylabel("CIFAR-10 test accuracy (%)")
ax[3].set_title("(d) Exact $M$ does not help", loc="left", fontsize=8.5, color=INK)

for a in ax:
    a.grid(color=GRID, lw=0.6, axis="y")
fig.tight_layout(w_pad=1.2, h_pad=1.4)
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/snowflake_diagnosis.pdf"); fig.savefig("results/figures/snowflake_diagnosis.png", dpi=200)
