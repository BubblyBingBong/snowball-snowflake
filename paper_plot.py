"""Report figure: train loss, val loss vs step, val loss vs wall-clock for MNIST and CIFAR-10 (3 seeds, mean + min/max).
Reads results/paper/{mnist,cifar10}.pt from paper_runs.py; writes results/paper/loss_curves.pdf (+ .png preview)."""
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

# fixed categorical order (dataviz reference palette slots 1-5); markers + dashes as secondary encoding
STYLE = [("adamw", "AdamW", "#2a78d6", "o", (0, (4, 1.5))), ("muon", "Muon", "#eb6834", "s", "-"),
         ("newton_muon", "Newton–Muon", "#1baf7a", "^", (0, (1.5, 1))),
         ("iam", "Snowball (ours)", "#eda100", "D", "-"), ("snowflake", "Snowflake (ours)", "#e87ba4", "v", (0, (6, 1.5, 1.5, 1.5)))]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8})


def smooth(y, k=101):
    t = torch.tensor(y, dtype=torch.float64).view(1, 1, -1)
    return F.avg_pool1d(t, k, 1, padding=k // 2, count_include_pad=False).view(-1)


def band(ax, x, ys, col, ls, mk, lab, every):
    Y = torch.stack([torch.as_tensor(y, dtype=torch.float64) for y in ys])
    ax.fill_between(x, Y.min(0).values, Y.max(0).values, color=col, alpha=0.18, lw=0)
    ax.plot(x, Y.mean(0), color=col, lw=1.3, ls=ls, marker=mk, ms=3.5, markevery=every, mec="white", mew=0.5, label=lab)


fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
for r, (ds, title) in enumerate([("mnist", "MNIST"), ("cifar10", "CIFAR-10")]):
    runs = torch.load(f"results/paper/{ds}.pt")["runs"]
    a0, a1, a2 = axes[r]
    for key, lab, col, mk, ls in STYLE:
        hs = runs[key]
        steps = hs[0]["step"]
        band(a0, steps, [smooth(h["train_loss"]) for h in hs], col, ls, mk, lab, len(steps) // 6)
        ev = hs[0]["eval_step"]
        band(a1, ev, [h["val_loss"] for h in hs], col, ls, mk, lab, max(1, len(ev) // 6))
        # wall-clock: cumulative measured step time (fwd+bwd+optimizer, eval excluded), mean over seeds, in seconds
        cum = torch.stack([torch.tensor(h["step_time"]).cumsum(0) / 1e3 for h in hs]).mean(0)
        band(a2, cum[torch.tensor(ev) - 1].tolist(), [h["val_loss"] for h in hs], col, ls, mk, lab, max(1, len(ev) // 6))
    a0.set_ylabel(f"{title}\ntrain loss (minibatch, smoothed)")
    a1.set_ylabel("validation loss"); a2.set_ylabel("validation loss")
    for a in (a0, a1, a2):
        a.set_yscale("log"); a.grid(color=GRID, lw=0.6, which="major")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    if ds == "mnist":
        lim, tk, tk0 = (0.04, 0.4), [0.05, 0.1, 0.2, 0.4], [1e-3, 1e-2, 1e-1, 1]
    else:
        lim, tk, tk0 = (1.1, 1.9), [1.1, 1.2, 1.4, 1.6, 1.8], [0.4, 0.6, 1, 2, 4]
    for a in (a1, a2):
        a.set_ylim(*lim)
    for a, t in ((a0, tk0), (a1, tk), (a2, tk)):
        a.yaxis.set_major_locator(FixedLocator(t)); a.yaxis.set_major_formatter(fmt); a.yaxis.set_minor_formatter(NullFormatter())
    a0.set_xlabel("step"); a1.set_xlabel("step"); a2.set_xlabel("wall-clock time (s)")
axes[0, 0].set_title("Training loss vs. step", loc="left", fontsize=8.5, color=INK)
axes[0, 1].set_title("Validation loss vs. step", loc="left", fontsize=8.5, color=INK)
axes[0, 2].set_title("Validation loss vs. time", loc="left", fontsize=8.5, color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5, handlelength=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=1.0)
fig.savefig("results/paper/loss_curves.pdf"); fig.savefig("results/paper/loss_curves.png", dpi=200)
