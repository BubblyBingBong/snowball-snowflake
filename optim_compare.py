"""AdamW vs Muon vs Newton-Muon vs Snowflake on a bias-free MLP (in -> 512 -> 512 -> 10).

Newton-Muon (Du & Su, arXiv 2604.01472, Alg. 1):
    K <- beta*K + (1-beta) * Z Z^T / N              (every k steps; K0 = 1e-3 I)
    Kinv <- (K + gamma * tr(K)/n * I)^-1
    G <- G Kinv   (raw gradient, before momentum)  -> standard Muon pipeline.
Snowflake (Input_Aware_Muon.pdf, Sec. 2.2):
    M = E[x x^T / ||x||^2]  (EMA, prior M0 = I/m = value for unit-Gaussian inputs, same refresh as above)
    dW = -eps * G Minv / sqrt(tr(G Minv G^T)),  Minv = (M + gamma*tr(M)/m * I)^-1,  eps = lr*sqrt(n_out/m_in)
    G here is the Nesterov-momentum gradient (same momentum as Muon); decoupled weight decay.

Usage: .venv/bin/python optim_compare.py --dataset mnist|cifar10 [--quick]
"""
import argparse, itertools, json, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
BATCH, HIDDEN, EVAL_EVERY = 128, 512, 50
OPTS = ["adamw", "muon", "newton_muon", "snowflake"]
COLORS = dict(adamw="C0", muon="C1", newton_muon="C3", snowflake="C2")


# ---------------------------------------------------------------- data
def load(name):
    if name == "mnist":
        tr, te = datasets.MNIST("data", train=True, download=True), datasets.MNIST("data", train=False, download=True)
        f = lambda d: ((d.data.float() / 255 - 0.1307) / 0.3081).reshape(len(d.data), -1)
        Xtr, Ytr, Xte, Yte = f(tr), tr.targets, f(te), te.targets
    else:
        # cached from huggingface.co/datasets/uoft-cs/cifar10 (uint8 NHWC), see README note in summary
        c = torch.load("data/cifar10.pt")
        mean, std = torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2470, 0.2435, 0.2616])
        f = lambda X: ((X.float() / 255 - mean) / std).permute(0, 3, 1, 2).reshape(len(X), -1)
        (Xtr, Ytr), (Xte, Yte) = (f(c["train"][0]), c["train"][1]), (f(c["test"][0]), c["test"][1])
    perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))
    va, trn = perm[:5000], perm[5000:]                     # 5k validation split held out of train
    d = lambda *t: tuple(x.to(DEV) for x in t)
    return d(Xtr[trn], Ytr[trn]), d(Xtr[va], Ytr[va]), d(Xte, Yte)


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, HIDDEN, bias=False), nn.Linear(HIDDEN, HIDDEN, bias=False),
                                     nn.Linear(HIDDEN, 10, bias=False)])
        self.inputs = [None] * 3                            # layer inputs from the latest forward pass

    def forward(self, x):
        for i, l in enumerate(self.layers):
            self.inputs[i] = x.detach()
            x = l(x)
            if i < 2:
                x = F.relu(x)
        return x


# ---------------------------------------------------------------- optimizers
@torch.no_grad()
def msgn(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if T else X


def damped_inverse(K, gamma):
    n = K.size(0)
    A = K + gamma * K.trace() / n * torch.eye(n, device=K.device, dtype=K.dtype)
    return torch.cholesky_inverse(torch.linalg.cholesky(A))


class MatrixOpt(torch.optim.Optimizer):
    """Muon / Newton-Muon / Snowflake.  `model.inputs[i]` must hold layer i's input for the current step."""
    def __init__(self, model, kind, lr, wd, gamma=0.2, beta=0.95, k=10, momentum=0.95):
        self.model, self.kind, self.beta, self.k, self.gamma, self.t = model, kind, beta, k, gamma, 0
        self.record = False   # if True, state[p]["update"] = this step's update without weight decay
        super().__init__([l.weight for l in model.layers], dict(lr=lr, weight_decay=wd, momentum=momentum))
        for p in self.param_groups[0]["params"]:
            m = p.size(1)
            st = self.state[p]
            st["buf"] = torch.zeros_like(p)
            if kind == "newton_muon":
                st["K"] = 1e-3 * torch.eye(m, device=p.device)
            elif kind == "snowflake":
                st["K"] = torch.eye(m, device=p.device) / m        # prior: unit-Gaussian inputs -> E[uu^T] = I/m
            if kind != "muon":
                st["Kinv"] = damped_inverse(st["K"], gamma)

    @torch.no_grad()
    def step(self):
        self.t += 1
        g = self.param_groups[0]
        for i, p in enumerate(g["params"]):
            st, G = self.state[p], p.grad
            n, m = p.shape
            if self.kind != "muon" and self.t % self.k == 0:      # refresh second-moment estimate + inverse
                Z = self.model.inputs[i]
                if self.kind == "snowflake":
                    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
                st["K"].mul_(self.beta).add_(Z.T @ Z / len(Z), alpha=1 - self.beta)
                st["Kinv"] = damped_inverse(st["K"], self.gamma)
            if self.kind == "newton_muon":
                G = G @ st["Kinv"]                                 # right-precondition the raw gradient
            buf = st["buf"].mul_(g["momentum"]).add_(G)
            u = G.add(buf, alpha=g["momentum"])                    # Nesterov
            p.mul_(1 - g["lr"] * g["weight_decay"])
            if self.kind == "snowflake":
                GM = u @ st["Kinv"]
                eps = g["lr"] * math.sqrt(n / m)
                scale = eps * (GM * u).sum().clamp_min(1e-30).rsqrt()     # tr(G Minv G^T) = <G Minv, G>; stays on GPU (no sync)
                upd = -GM * scale
                if self.record: st["update"], st["eps"] = upd.clone(), eps
                p.add_(upd)
            else:
                upd = msgn(u) * (-g["lr"] * max(1, n / m) ** 0.5)
                if self.record: st["update"] = upd.clone()
                p.add_(upd)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, X, Y):
    out = torch.cat([model(X[i:i + 5000]) for i in range(0, len(X), 5000)])
    return F.cross_entropy(out, Y).item(), (out.argmax(1) == Y).float().mean().item()


def train(kind, cfg, data, epochs, seed=0, callback=None, record=False, make_opt=None):
    """callback(step, total, model, opt) is called right after opt.step(), before the LR scheduler advances.
    make_opt(model) -> optimizer overrides the built-in optimizer construction (used by snowflake_oracle.py)."""
    (Xtr, Ytr), (Xva, Yva), _ = data
    torch.manual_seed(seed)
    model = MLP(Xtr.size(1)).to(DEV)
    if make_opt is not None:
        opt = make_opt(model)
    elif kind == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = MatrixOpt(model, kind, cfg["lr"], cfg["wd"], gamma=cfg.get("gamma", 0.2))
        opt.record = record
    spe = len(Xtr) // BATCH
    total = epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / total)
    h = dict(step=[], train_loss=[], step_time=[], eval_step=[], val_loss=[], val_acc=[])
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEV)
        for b in range(spe):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            torch.cuda.synchronize(); t0 = time.perf_counter()   # time per step = fwd + bwd + optimizer (no eval)
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); step += 1
            if callback is not None:
                callback(step, total, model, opt)
            sched.step()
            torch.cuda.synchronize(); h["step_time"].append(1e3 * (time.perf_counter() - t0))
            h["step"].append(step); h["train_loss"].append(loss.item())
            if step % EVAL_EVERY == 0 or step == total:
                vl, va = evaluate(model, Xva, Yva)
                h["eval_step"].append(step); h["val_loss"].append(vl); h["val_acc"].append(va)
                if not math.isfinite(vl):
                    return model, h
    return model, h


def smooth(y, k=25):
    t = torch.tensor(y)
    if len(t) < k: return y
    return F.avg_pool1d(t.view(1, 1, -1), k, 1, padding=k // 2, count_include_pad=False).view(-1)[:len(y)].tolist()


def cfg_str(c):
    return ", ".join(f"{k}={v:g}" for k, v in c.items())


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--sweep-only", action="store_true", help="run/cache the sweep, skip final runs")
    ap.add_argument("--quick", action="store_true", help="1 config per optimizer, 1 epoch (smoke test)")
    args = ap.parse_args()
    epochs = 1 if args.quick else dict(mnist=5, cifar10=10)[args.dataset]
    out = f"results/optim_compare/{args.dataset}" + ("_quick" if args.quick else "")
    os.makedirs(out, exist_ok=True)
    data = load(args.dataset)

    # v2 grids: widened after v1 picked edge values (CIFAR overfits, so final val loss favors strong lr*wd)
    LRS_M = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    grids = dict(
        adamw=dict(lr=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], wd=[0.0, 0.1, 1.0]),
        muon=dict(lr=LRS_M, wd=[0.0, 0.1, 0.3]),
        newton_muon=dict(lr=[0.002] + LRS_M, wd=[0.1, 0.3], gamma=[0.2, 1.0, 5.0]),
        snowflake=dict(lr=LRS_M, wd=[0.1, 0.3], gamma=[0.2, 1.0, 5.0, 20.0]),
    )
    if args.quick:
        grids = dict(adamw=dict(lr=[1e-3], wd=[0.0]), muon=dict(lr=[0.02], wd=[0.1]),
                     newton_muon=dict(lr=[0.02], wd=[0.1], gamma=[0.2]), snowflake=dict(lr=[0.02], wd=[0.1], gamma=[0.2]))

    summary, best = {}, {}
    for kind in OPTS:
        keys = list(grids[kind])
        runs = []
        t0 = time.time()
        for vals in itertools.product(*grids[kind].values()):
            cfg = dict(zip(keys, vals))
            cache = f"{out}/runs/{kind}__{cfg_str(cfg).replace(', ', '_')}.pt"
            if os.path.exists(cache):
                h = torch.load(cache)
            else:
                _, h = train(kind, cfg, data, epochs)
                os.makedirs(f"{out}/runs", exist_ok=True); torch.save(h, cache)
            vl = h["val_loss"][-1] if math.isfinite(h["val_loss"][-1]) else float("inf")
            runs.append((cfg, h, vl))
            print(f"[{args.dataset}] {kind:12s} {cfg_str(cfg):32s} val_loss={vl:.4f} val_acc={h['val_acc'][-1]*100:.2f}%", flush=True)
        runs.sort(key=lambda r: r[2])
        best[kind] = runs[0][0]
        print(f"[{args.dataset}] BEST {kind}: {cfg_str(best[kind])}  ({time.time()-t0:.0f}s)", flush=True)
        summary[kind] = dict(best=best[kind], sweep=[dict(cfg=c, val_loss=v, val_acc=h["val_acc"][-1]) for c, h, v in runs])

        # sweep plot (val loss of every config, best in bold)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_prop_cycle(color=plt.cm.tab20.colors)
        for c, h, v in runs:
            bold = c == best[kind]
            ax.plot(h["eval_step"], h["val_loss"], lw=2.5 if bold else 0.8, alpha=1 if bold else 0.5,
                    label=cfg_str(c) if bold or len(runs) <= 12 else None, color="k" if bold else None)
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.grid(alpha=.3)
        ax.set_title(f"{args.dataset}: {kind} sweep ({len(runs)} configs; black = best)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{out}/sweep_{kind}.png", dpi=130); plt.close(fig)

    if args.sweep_only:
        json.dump(summary, open(f"{out}/summary_sweep.json", "w"), indent=2)
        raise SystemExit(0)
    # final: rerun best configs, report train/val/test, plot curves
    final = {}
    for kind in OPTS:
        model, h = train(kind, best[kind], data, epochs)
        trl, tra = evaluate(model, *data[0]); tel, tea = evaluate(model, *data[2])
        final[kind] = h
        summary[kind]["final"] = dict(train_loss=trl, train_acc=tra, val_loss=h["val_loss"][-1], val_acc=h["val_acc"][-1],
                                      test_loss=tel, test_acc=tea)
        print(f"[{args.dataset}] FINAL {kind:12s} {cfg_str(best[kind]):32s} train_loss={trl:.4f} val_loss={h['val_loss'][-1]:.4f} "
              f"test_loss={tel:.4f} test_acc={tea*100:.2f}%  median step {torch.tensor(h['step_time']).median():.2f} ms", flush=True)
        summary[kind]["final"]["median_step_ms"] = torch.tensor(h["step_time"]).median().item()
        summary[kind]["final"]["mean_step_ms"] = sum(h["step_time"]) / len(h["step_time"])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for kind, h in final.items():
        lab = f"{kind} ({cfg_str(best[kind])})"
        ax[0].plot(h["step"], smooth(h["train_loss"]), color=COLORS[kind], label=lab)
        ax[1].plot(h["eval_step"], h["val_loss"], color=COLORS[kind], label=lab)
        ax[2].plot(h["eval_step"], [100 * a for a in h["val_acc"]], color=COLORS[kind], label=lab)
    for a, t in zip(ax, ["train loss (smoothed)", "val loss", "val accuracy (%)"]):
        a.set_xlabel("step"); a.set_ylabel(t); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    accs = [100 * a for h in final.values() for a in h["val_acc"]]
    ax[2].set_ylim(max(accs) - 8, max(accs) + 1)
    fig.suptitle(f"{args.dataset}: best config of each optimizer (selected by final val loss)")
    fig.tight_layout(); fig.savefig(f"{out}/best_curves.png", dpi=130); plt.close(fig)

    # step vs time per step (best config of each optimizer)
    fig, ax = plt.subplots(figsize=(10, 5))
    for kind, h in final.items():
        ax.plot(h["step"], h["step_time"], color=COLORS[kind], lw=0.4, alpha=0.3)
        med = torch.tensor(h["step_time"]).median().item()
        ax.plot(h["step"], smooth(h["step_time"], 51), color=COLORS[kind], lw=2, label=f"{kind}  (median {med:.2f} ms)")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("time per step (ms, log)"); ax.grid(alpha=.3, which="both")
    ax.set_title(f"{args.dataset}: wall-clock time per training step (fwd+bwd+optimizer; thin = raw, thick = smoothed)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{out}/step_time.png", dpi=130); plt.close(fig)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    print("saved", out)
