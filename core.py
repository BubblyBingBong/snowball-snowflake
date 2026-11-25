"""Shared code: data, model, training loop, and the optimizers (Muon, Newton-Muon, Snowflake, Snowball).

Model: bias-free MLP d_in -> 512 -> 512 -> 10 (ReLU). Batch 128, linear learning-rate decay to 0.

Newton-Muon (Du & Su, arXiv 2604.01472, Alg. 1):
    K <- beta*K + (1-beta) * Z Z^T / N    (every k steps; K0 = 1e-3 I)
    Kinv <- (K + gamma * tr(K)/n * I)^-1
    G <- G Kinv   (raw gradient, before momentum)  -> standard Muon pipeline.
Snowflake:
    M = E[x x^T / ||x||^2]  (EMA, prior M0 = I/m, same refresh as above)
    dW = -eps * G Minv / sqrt(tr(G Minv G^T)),  Minv = (M + gamma*tr(M)/m * I)^-1,  eps = lr*sqrt(n_out/m_in)
    G is the Nesterov-momentum gradient (same momentum as Muon); decoupled weight decay.
Snowball:
    dW = -lr * sqrt(max(1, n/m)) * orth(G C^-1/2) C^-1/2,  C = M damped as above and rescaled to trace m
    (gamma -> inf recovers Muon exactly). G is the Nesterov-momentum gradient.
"""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
BATCH, HIDDEN, EVAL_EVERY = 128, 512, 50


# ---------------------------------------------------------------- data
def load(name):
    if name == "mnist":
        tr, te = datasets.MNIST("data", train=True, download=True), datasets.MNIST("data", train=False, download=True)
        f = lambda d: ((d.data.float() / 255 - 0.1307) / 0.3081).reshape(len(d.data), -1)
        Xtr, Ytr, Xte, Yte = f(tr), tr.targets, f(te), te.targets
    else:
        # written by prepare_data.py from huggingface.co/datasets/uoft-cs/cifar10 (uint8 NHWC)
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
    make_opt(model) -> optimizer overrides the built-in optimizer construction (Snowball, oracle variants)."""
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


def cfg_str(c):
    return ", ".join(f"{k}={v:g}" for k, v in c.items())


# ---------------------------------------------------------------- Snowball
def mpow(A, p):                                                   # symmetric PSD matrix power
    ev, V = torch.linalg.eigh(A)
    return (V * ev.clamp_min(1e-12).pow(p)) @ V.T


def metric(K, gamma):                                             # damped, rescaled to trace d
    d = K.size(0)
    A = K + gamma * K.trace() / d * torch.eye(d, device=K.device)
    return A * (d / A.trace())


class Snowball(MatrixOpt):
    """Muon in whitened input coordinates: orth(G C^-1/2) C^-1/2 with C = metric(M, gamma), refreshed every k steps."""
    def __init__(self, model, lr, wd, gamma):
        super().__init__(model, "snowflake", lr, wd, gamma=gamma)     # reuses the EMA of M (prior I/m)
        for p in self.param_groups[0]["params"]:
            self.state[p]["Cp"] = torch.eye(p.size(1), device=p.device)   # C^-1/2

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
                st["Cp"] = mpow(metric(st["K"], self.gamma), -0.5)
            buf = st["buf"].mul_(g["momentum"]).add_(G)
            u = G.add(buf, alpha=g["momentum"])
            p.mul_(1 - g["lr"] * g["weight_decay"])
            p.sub_(g["lr"] * max(1, n / m) ** 0.5 * (msgn(u @ st["Cp"]) @ st["Cp"]))


def make_snowball(cfg):
    return lambda model: Snowball(model, cfg["lr"], cfg["wd"], cfg["gamma"])
