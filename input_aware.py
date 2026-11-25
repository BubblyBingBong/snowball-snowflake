"""Input-aware Muon variants on the MLPs (MNIST / CIFAR-10), compared with cached Muon / Newton-Muon / Snowflake.

All variants measure inputs in a damped, trace-normalized metric C = norm(M + gamma tr(M)/m I), M = E[u u^T],
u = x/||x|| (same EMA statistics as Snowflake). Trace normalization (tr C = m) makes gamma -> inf recover Muon exactly.

  iamuon     (Input_Aware_Muon.pdf Sec. 1)  max ||dW x|| / ||C^-1/2 x||  ->  dW = -s * orth(G C^-1/2) C^-1/2
  partial    same with C^-1/4 on both sides of orth (half-way whitening)
  twosided   input metric C and output metric S (S = E[d d^T], d = dLoss/d output, damped+trace-normalized):
             dW = -s * S^-1/2 orth(S^-1/2 G C^-1/2) C^-1/2
  schatten4  Schatten-4 norm of dW C^1/2 (between Snowflake = Schatten-2 and iamuon = Schatten-inf):
             G C^-1/2 = U Sig V^T  ->  dW = -s * U (Sig/||Sig||_inf)^(1/3) V^T C^-1/2
s = lr * max(1, n/m)^0.5 (Muon's scale); G = Nesterov momentum gradient; decoupled weight decay.

Usage: .venv/bin/python input_aware.py --dataset mnist|cifar10 [--methods iamuon,partial,...]
"""
import argparse, itertools, json, math, os
import torch
from optim_compare import load, train, evaluate, MatrixOpt, msgn, cfg_str


def mpow(A, p):                                                   # symmetric PSD matrix power
    ev, V = torch.linalg.eigh(A)
    return (V * ev.clamp_min(1e-12).pow(p)) @ V.T


def metric(K, gamma):
    d = K.size(0)
    A = K + gamma * K.trace() / d * torch.eye(d, device=K.device)
    return A * (d / A.trace())


class InputAware(MatrixOpt):
    def __init__(self, model, method, lr, wd, gamma, gamma_s=0.1):
        super().__init__(model, "snowflake", lr, wd, gamma=gamma)     # reuses K (EMA of M) state, prior I/m
        self.method, self.gamma_s = method, gamma_s
        self.alpha = 0.25 if method == "partial" else 0.5
        self.deltas = [None] * len(model.layers)
        for i, l in enumerate(model.layers):
            p = l.weight; st = self.state[p]
            st["Cp"] = torch.eye(p.size(1), device=p.device)          # C^-alpha
            if method == "twosided":
                st["S"] = torch.zeros(p.size(0), p.size(0), device=p.device); st["St"] = 0
                st["Sp"] = torch.eye(p.size(0), device=p.device)      # S^-1/2
                l.register_forward_hook(self._fwd(i))

    def _fwd(self, i):
        def hook(mod, inp, out):
            if out.requires_grad:
                out.register_hook(lambda g: self.deltas.__setitem__(i, g.detach()))
        return hook

    @torch.no_grad()
    def step(self):
        self.t += 1
        g = self.param_groups[0]
        for i, p in enumerate(g["params"]):
            st, G = self.state[p], p.grad
            n, m = p.shape
            if self.t % self.k == 0:
                Z = self.model.inputs[i]; Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
                st["K"].mul_(self.beta).add_(Z.T @ Z / len(Z), alpha=1 - self.beta)
                st["Cp"] = mpow(metric(st["K"], self.gamma), -self.alpha)
                if self.method == "twosided":
                    D = self.deltas[i] * len(self.deltas[i])
                    st["S"].mul_(self.beta).add_(D.T @ D / len(D), alpha=1 - self.beta); st["St"] += 1
                    st["Sp"] = mpow(metric(st["S"] / (1 - self.beta ** st["St"]), self.gamma_s), -0.5)
            buf = st["buf"].mul_(g["momentum"]).add_(G)
            u = G.add(buf, alpha=g["momentum"])
            p.mul_(1 - g["lr"] * g["weight_decay"])
            s = g["lr"] * max(1, n / m) ** 0.5
            Cp = st["Cp"]
            if self.method in ("iamuon", "partial"):
                upd = msgn(u @ Cp) @ Cp
            elif self.method == "twosided":
                upd = st["Sp"] @ msgn(st["Sp"] @ u @ Cp) @ Cp
            elif self.method == "schatten4":
                X = u @ Cp                                       # U Sig^(1/3) V^T = (X X^T)^(-1/3) X, via n x n eigh
                ev, V = torch.linalg.eigh(X @ X.T)
                ev = ev.clamp_min(ev[-1] * 1e-12 + 1e-30)
                upd = ((V * (ev / ev[-1]).pow(-1 / 3)) @ V.T @ X / ev[-1].sqrt()) @ Cp
            p.sub_(s * upd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--methods", default="iamuon,partial,twosided,schatten4")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    EPOCHS = dict(mnist=5, cifar10=10)[args.dataset]
    OUT = f"results/input_aware/{args.dataset}"; os.makedirs(f"{OUT}/runs", exist_ok=True)
    data = load(args.dataset)
    prev = json.load(open(f"results/optim_compare/{args.dataset}/summary.json"))
    wd = prev["muon"]["best"]["wd"]
    mk = lambda meth, c: (lambda model: InputAware(model, meth, c["lr"], c["wd"], c["gamma"]))
    best = {}
    for meth in args.methods.split(","):
        rows = []
        for lr, gm in itertools.product([0.005, 0.01, 0.02, 0.05, 0.1], [0.05, 0.2, 1.0, 5.0]):
            c = dict(lr=lr, wd=wd, gamma=gm)
            f = f"{OUT}/runs/{meth}__{cfg_str(c).replace(', ', '_')}.pt"
            if os.path.exists(f): h = torch.load(f)
            else: _, h = train("snowflake", c, data, EPOCHS, make_opt=mk(meth, c)); torch.save(h, f)
            vl = h["val_loss"][-1] if math.isfinite(h["val_loss"][-1]) else float("inf")
            rows.append((c, vl, h["val_acc"][-1]))
            print(f"[{args.dataset}] {meth:10s} {cfg_str(c):30s} val_loss={vl:.4f} val_acc={h['val_acc'][-1]*100:.2f}%", flush=True)
        rows.sort(key=lambda r: r[1]); best[meth] = rows[0][0]
        print(f"[{args.dataset}] BEST {meth}: {cfg_str(best[meth])}  val_loss {rows[0][1]:.4f}", flush=True)

    final = {}
    runs = [("muon", lambda s: train("muon", prev["muon"]["best"], data, EPOCHS, seed=s))]
    runs += [(m, (lambda m: lambda s: train("snowflake", best[m], data, EPOCHS, seed=s, make_opt=mk(m, best[m])))(m)) for m in best]
    for name, run in runs:
        final[name] = []
        for s in range(args.seeds):
            model, h = run(s); tl, ta = evaluate(model, *data[2])
            final[name].append(dict(seed=s, val_loss=h["val_loss"][-1], test_acc=ta,
                                    train_loss_end=sum(h["train_loss"][-100:]) / 100))
    print(f"\n[{args.dataset}] mean over {args.seeds} seeds:")
    for k, v in final.items():
        ta = torch.tensor([r["test_acc"] * 100 for r in v]); vl = torch.tensor([r["val_loss"] for r in v])
        tr = torch.tensor([r["train_loss_end"] for r in v])
        print(f"  {k:10s} {cfg_str(best.get(k, prev['muon']['best'])):30s} test acc {ta.mean():.2f} ± {ta.std():.2f}%   "
              f"val loss {vl.mean():.4f}   train loss {tr.mean():.4f}", flush=True)
    json.dump(dict(best=best, final=final), open(f"{OUT}/summary.json", "w"), indent=1)
