"""Report figure for Sec. 7.1 (diagnosing Snowflake), from saved results only:
 (a) Muon/AdamW: mean effective scaling / spectral norm of each update over training (MNIST, results/final_histories.pt)
 (b) CIFAR-10 Snowflake: best val loss vs damping gamma, estimated vs exact M (results/snowflake_oracle sweeps)
 (c) CIFAR-10 test accuracy, 3 seeds: Snowflake estimated / exact / M=I vs Muon (results/snowflake_oracle final)
-> results/paper/snowflake_diagnosis.pdf (+ .png)"""
import json
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

MUON, ADAMW, SF, SFX, NEUTRAL = "#eb6834", "#2a78d6", "#e87ba4", "#6250d6", "#8f8e86"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8})


def smooth(y, k=51):
    t = torch.tensor(y, dtype=torch.float64).view(1, 1, -1)
    return F.avg_pool1d(t, k, 1, padding=k // 2, count_include_pad=False).view(-1)


fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

# (a) hypothesis 1
h = torch.load("results/final_histories.pt", weights_only=False)
ax[0].plot(h["muon"]["step"], smooth(h["muon"]["ratio"]), color=MUON, lw=1.4)
ax[0].axhline(1.0, color=MUTED, lw=0.9, ls=":")
ax[0].annotate("inputs along top singular direction", (2150, 1.0), xytext=(0, -9), textcoords="offset points",
               ha="right", fontsize=7, color=MUTED)
ax[0].set_ylim(0, 1.1); ax[0].set_xlabel("step (MNIST)")
ax[0].set_ylabel("effective scaling / spectral norm")
ax[0].set_title("(a) Slack exists", loc="left", fontsize=8.5, color=INK)
ax[0].annotate("Muon updates", (1100, 0.66), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK)

# (b) hypothesis 2: damping preference with estimated vs exact M
s = json.load(open("results/snowflake_oracle/cifar10/summary.json"))
for var, lab, col, mk, ls in [("est", "estimated $M$", SF, "o", (0, (4, 1.5))), ("oracle", "exact $M$", SFX, "s", "-")]:
    best = {}
    for r in s["sweeps"][var]:
        g = r["cfg"]["gamma"]; best[g] = min(best.get(g, 9.0), r["val_loss"])
    if var == "est":                           # extended damping grid (gamma 50, 100) from ext_damping.py
        ext = json.load(open("results/paper/extended_damping.json"))["cifar10"]["snowflake"]["best_val_loss_by_gamma"]
        best.update({float(g): v for g, v in ext.items()})
    gs = sorted(best)
    ax[1].plot(gs, [best[g] for g in gs], color=col, lw=1.4, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, label=lab)
GINF = 200.0                                   # plotted position of gamma = infinity (M = I)
ident = min(r["val_loss"] for r in s["sweeps"]["identity"])
ax[1].plot([20.0, GINF], [best[20.0], ident], color=SFX, lw=1.0, ls=":")
ax[1].plot([GINF], [ident], marker="D", ms=4.5, color=NEUTRAL, mec="white", mew=0.6, ls="none", label="$M = I$ ($\\gamma=\\infty$)")
mu = s["table"]["muon"]["val_loss_mean"]
ax[1].axhline(mu, color=MUON, lw=1.0, ls=":")
ax[1].annotate("Muon", (0.012, mu), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK)
ax[1].set_xscale("log"); ax[1].set_xlabel(r"damping $\gamma$ (CIFAR-10)"); ax[1].set_ylabel("best validation loss")
ax[1].xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, GINF]))
ax[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$\infty$" if v == GINF else f"{v:g}")); ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_title("(b) Damping preference", loc="left", fontsize=8.5, color=INK)
ax[1].legend(frameon=False, fontsize=7, loc="upper right")

# (c) test accuracy, 3 seeds
order = [("est", "est. $M$", SF), ("oracle", "exact $M$", SFX),
         ("identity", "$M = I$", NEUTRAL), ("muon", "Muon", MUON)]
for i, (k, lab, col) in enumerate(order):
    accs = [100 * r["test_acc"] for r in s["final"][k]]
    m = sum(accs) / len(accs)
    ax[2].bar(i, m - 55, bottom=55, width=0.62, color=col, edgecolor="white", linewidth=1)
    ax[2].scatter([i] * len(accs), accs, s=7, color=INK, zorder=3, linewidths=0)
    ax[2].annotate(f"{m:.1f}", (i, max(accs)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK)
ax[2].set_xticks(range(len(order))); ax[2].set_xticklabels([o[1] for o in order], fontsize=7)
ax[2].set_xlabel("Snowflake variants vs. Muon")
ax[2].set_ylim(55, 60.6); ax[2].set_ylabel("CIFAR-10 test accuracy (%)")
ax[2].set_title("(c) Exact $M$ does not help", loc="left", fontsize=8.5, color=INK)

for a in ax:
    a.grid(color=GRID, lw=0.6, axis="y")
fig.tight_layout(w_pad=1.2)
fig.savefig("results/paper/snowflake_diagnosis.pdf"); fig.savefig("results/paper/snowflake_diagnosis.png", dpi=200)
