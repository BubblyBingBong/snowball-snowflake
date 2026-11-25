"""MLP on MNIST: AdamW vs Muon, hyperparameter sweeps, and a per-step
effective-norm / spectral-norm analysis of Muon's weight updates.

Usage:  .venv/bin/python mnist_muon.py
Outputs go to ./results/
"""
import itertools, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets

torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 for the norm analysis
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

EPOCHS, BATCH, HIDDEN = 5, 128, 512
EVAL_EVERY = 50  # steps


# ---------------------------------------------------------------- data
def load_mnist():
    tr = datasets.MNIST("data", train=True, download=True)
    te = datasets.MNIST("data", train=False, download=True)
    norm = lambda x: ((x.float() / 255.0 - 0.1307) / 0.3081).view(len(x), -1).to(DEV)
    X, Y = norm(tr.data), tr.targets.to(DEV)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g).to(DEV)
    tr_idx, va_idx = perm[:55000], perm[55000:]          # 55k train / 5k val
    return (X[tr_idx], Y[tr_idx]), (X[va_idx], Y[va_idx]), (norm(te.data), te.targets.to(DEV))


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    # bias-free so every parameter is a weight matrix (all trained by Muon)
    def __init__(self, dims=(784, HIDDEN, HIDDEN, 10)):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------- Muon
@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration approximating the orthogonal factor U V^T of G
    (coefficients from Keller Jordan's Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                o = newton_schulz(u, g["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"] * scale)


def make_opt(name, model, lr, wd):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return Muon(model.parameters(), lr=lr, weight_decay=wd)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    logits = model(X)
    model.train()
    return F.cross_entropy(logits, Y).item(), (logits.argmax(1) == Y).float().mean().item()


def train(opt_name, lr, wd, data, seed=0, analyze=False, log=True, snapshot_steps=(), dims=None):
    """snapshot_steps: steps at which to save (dW, x, W_before) for every layer into hist['snapshots'][step]."""
    (Xtr, Ytr), (Xva, Yva), _ = data
    torch.manual_seed(seed)
    model = (MLP(dims) if dims else MLP()).to(DEV)
    opt = make_opt(opt_name, model, lr, wd)
    steps_per_epoch = len(Xtr) // BATCH
    total = EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / total)  # linear decay to 0

    # capture each layer's input x (y = W x) for the analysis
    inputs = {}
    hooks = []
    if analyze:
        for i, l in enumerate(model.layers):
            hooks.append(l.register_forward_hook(lambda m, a, o, i=i: inputs.__setitem__(i, a[0].detach())))

    hist = dict(step=[], train_loss=[], eval_step=[], val_loss=[], val_acc=[], ratio=[], ratio_per_layer=[], snapshots={})
    step = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), device=DEV)
        for b in range(steps_per_epoch):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if analyze:
                W_before = [l.weight.detach().clone() for l in model.layers]
            opt.step()
            sched.step()
            step += 1
            hist["step"].append(step)
            hist["train_loss"].append(loss.item())

            if analyze:
                with torch.no_grad():
                    per_layer = []
                    for i, l in enumerate(model.layers):
                        dW = l.weight - W_before[i]                       # (out, in)
                        x = inputs[i]                                     # (B, in)
                        spec = torch.linalg.matrix_norm(dW, ord=2)        # sigma_max(dW)
                        eff = (x @ dW.T).norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)  # ||dW x||/||x|| per sample
                        per_layer.append((eff / spec).mean().item())      # avg over batch
                    if step in snapshot_steps:
                        hist["snapshots"][step] = [((l.weight - W_before[i]).cpu(), inputs[i].cpu(), W_before[i].cpu())
                                                   for i, l in enumerate(model.layers)]
                    hist["ratio_per_layer"].append(per_layer)
                    hist["ratio"].append(sum(per_layer) / len(per_layer))  # avg over layers

            if step % EVAL_EVERY == 0 or step == total:
                vl, va = evaluate(model, Xva, Yva)
                hist["eval_step"].append(step)
                hist["val_loss"].append(vl)
                hist["val_acc"].append(va)
                if not (vl == vl):  # NaN -> diverged, stop early
                    break
        if hist["val_loss"] and hist["val_loss"][-1] != hist["val_loss"][-1]:
            break
    for h in hooks:
        h.remove()
    if log:
        print(f"  {opt_name:5s} lr={lr:<7g} wd={wd:<5g}  val_loss={hist['val_loss'][-1]:.4f}  "
              f"val_acc={hist['val_acc'][-1]*100:.2f}%", flush=True)
    return model, hist


# ---------------------------------------------------------------- plotting
def smooth(y, k=25):
    t = torch.tensor(y)
    if len(t) < k:
        return y
    return torch.nn.functional.avg_pool1d(t.view(1, 1, -1), k, 1, padding=k // 2, count_include_pad=False).view(-1)[:len(y)].tolist()


def plot_sweep(name, runs, best_key):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a_ in ax: a_.set_prop_cycle(color=plt.cm.tab20.colors)
    for key, h in runs.items():
        lw, a = (2.5, 1.0) if key == best_key else (1.0, 0.55)
        line, = ax[0].plot(h["step"], smooth(h["train_loss"]), lw=lw, alpha=a, label=f"lr={key[0]:g}, wd={key[1]:g}")
        ax[1].plot(h["eval_step"], h["val_loss"], lw=lw, alpha=a, color=line.get_color())
    for a_, t in zip(ax, ["train loss (smoothed)", "val loss"]):
        a_.set_yscale("log"); a_.set_xlabel("step"); a_.set_ylabel(t); a_.grid(alpha=0.3)
    ax[0].set_title(f"{name} sweep: train loss  (bold = best)")
    ax[1].set_title(f"{name} sweep: val loss")
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/sweep_{name}.png", dpi=130); plt.close(fig)


def plot_best(best):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, (cfg, h) in best.items():
        lab = f"{name} (lr={cfg[0]:g}, wd={cfg[1]:g})"
        ax[0].plot(h["step"], smooth(h["train_loss"]), label=lab)
        ax[1].plot(h["eval_step"], h["val_loss"], label=lab)
        ax[2].plot(h["eval_step"], [100 * v for v in h["val_acc"]], label=lab)
    for a_, t in zip(ax, ["train loss (smoothed)", "val loss", "val accuracy (%)"]):
        a_.set_xlabel("step"); a_.set_ylabel(t); a_.grid(alpha=0.3); a_.legend()
    ax[0].set_yscale("log"); ax[1].set_yscale("log"); ax[2].set_ylim(96, 100)
    fig.suptitle("Best AdamW vs best Muon"); fig.tight_layout()
    fig.savefig(f"{OUT}/best_adamw_vs_muon.png", dpi=130); plt.close(fig)


def plot_ratio(h_muon, h_adam):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].plot(h_muon["step"], h_muon["ratio"], lw=0.6, alpha=0.35, color="C1")
    ax[0].plot(h_muon["step"], smooth(h_muon["ratio"], 51), lw=2, color="C1", label="Muon")
    if h_adam is not None:
        ax[0].plot(h_adam["step"], h_adam["ratio"], lw=0.6, alpha=0.35, color="C0")
        ax[0].plot(h_adam["step"], smooth(h_adam["ratio"], 51), lw=2, color="C0", label="AdamW (reference)")
    ax[0].set_title(r"mean over layers & batch of  $\frac{\|\Delta W x\|/\|x\|}{\|\Delta W\|_2}$")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("effective / spectral norm"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_ylim(0, 1)
    per = torch.tensor(h_muon["ratio_per_layer"])
    for i in range(per.shape[1]):
        ax[1].plot(h_muon["step"], smooth(per[:, i].tolist(), 51), label=f"layer {i+1} {tuple(MLP().layers[i].weight.shape)}")
    ax[1].set_title("Muon: per-layer ratio (smoothed)"); ax[1].set_xlabel("step"); ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/muon_effective_vs_spectral_ratio.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    data = load_mnist()
    grids = {
        "adamw": dict(lr=[3e-4, 1e-3, 3e-3, 1e-2], wd=[0.0, 0.01, 0.1]),
        "muon":  dict(lr=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], wd=[0.0, 0.01, 0.1]),
    }
    summary, best = {}, {}
    for name, grid in grids.items():
        print(f"== {name} sweep ({EPOCHS} epochs, batch {BATCH}, selection on 5k val split)")
        runs = {}
        t0 = time.time()
        for lr, wd in itertools.product(grid["lr"], grid["wd"]):
            _, runs[(lr, wd)] = train(name, lr, wd, data)
        score = lambda k: runs[k]["val_loss"][-1] if runs[k]["val_loss"][-1] == runs[k]["val_loss"][-1] else 1e9
        best_key = min(runs, key=score)
        print(f"  best {name}: lr={best_key[0]} wd={best_key[1]}  ({time.time()-t0:.0f}s)")
        plot_sweep(name, runs, best_key)
        summary[name] = {"sweep": {f"lr={k[0]},wd={k[1]}": dict(val_loss=v["val_loss"][-1], val_acc=v["val_acc"][-1])
                                   for k, v in runs.items()},
                         "best": dict(lr=best_key[0], wd=best_key[1])}
        best[name] = best_key

    # final runs with best settings: test-set numbers + Muon update analysis (AdamW too, as a reference)
    final = {}
    for name in ["adamw", "muon"]:
        lr, wd = best[name]
        model, h = train(name, lr, wd, data, analyze=True, log=False)
        tl, ta = evaluate(model, *data[2])
        trl, tra = evaluate(model, *data[0])
        final[name] = ((lr, wd), h)
        summary[name]["final"] = dict(train_loss=trl, train_acc=tra, val_loss=h["val_loss"][-1],
                                      val_acc=h["val_acc"][-1], test_loss=tl, test_acc=ta,
                                      mean_ratio=sum(h["ratio"]) / len(h["ratio"]))
        print(f"FINAL {name}: lr={lr} wd={wd}  train_loss={trl:.4f}  test_loss={tl:.4f}  "
              f"test_acc={ta*100:.2f}%  mean eff/spec ratio={summary[name]['final']['mean_ratio']:.3f}")
    plot_best(final)
    plot_ratio(final["muon"][1], final["adamw"][1])
    torch.save({k: v[1] for k, v in final.items()}, f"{OUT}/final_histories.pt")
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print("saved plots + summary to", OUT)
