"""Which update DIRECTION is best for one step, given the same batch gradient and a perfect second moment M?

Checkpoints: the best normal-Snowflake config is trained and frozen at a few steps. At each checkpoint, for B random
batches of 128, take the batch gradient G_l of every layer (no momentum) and build candidate directions:
    snowflake(gamma)   G (M + gamma tr(M)/m I)^-1          gamma in {0.01, 0.2, 1, 5, 20}
    normalized GD      G                                   (= gamma -> infinity)
    muon               orth(G) = U V^T
    newton-muon        orth(G (M + 0.2 tr(M)/m I)^-1)
M is EXACT (full training set, current weights). Each direction is scaled per layer to size c * sqrt(n/m), where
size is either (a) RMS effective scaling sqrt(tr(D M D^T)) or (b) spectral norm, and c sweeps a log grid. We record
the change in FULL training loss after the step and report the best decrease each direction can reach (own best c).

Also: per-eigendirection gradient signal-to-noise for layer 1:  signal_k = ||G_full v_k||^2,
noise_k = E_batch ||(G_b - G_full) v_k||^2, v_k = eigenvectors of M.

Usage: .venv/bin/python snowflake_direction.py --dataset mnist|cifar10
"""
import argparse, copy, json, os
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optim_compare import load, train, msgn, damped_inverse, BATCH


@torch.no_grad()
def norm_second_moment(Z, chunk=10000):                 # E[u u^T], u = z/||z||, over all rows of Z
    M = torch.zeros(Z.size(1), Z.size(1), device=Z.device)
    for i in range(0, len(Z), chunk):
        U = Z[i:i + chunk]; U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        M += U.T @ U
    return M / len(Z)


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
ap.add_argument("--batches", type=int, default=16)
args = ap.parse_args()
OUT = f"results/snowflake_direction/{args.dataset}"
os.makedirs(OUT, exist_ok=True)
data = load(args.dataset)
(Xtr, Ytr), _, _ = data
cfg = json.load(open(f"results/optim_compare/{args.dataset}/summary.json"))["snowflake"]["best"]
EPOCHS = dict(mnist=5, cifar10=10)[args.dataset]
total = EPOCHS * (len(Xtr) // BATCH)
CKPTS = [100, total // 2, total - 1]

snaps = {}
def cb(step, tot, model, opt):
    if step in CKPTS: snaps[step] = copy.deepcopy(model.state_dict())
train("snowflake", cfg, data, EPOCHS, callback=cb)


@torch.no_grad()
def full_loss(model):
    return sum(F.cross_entropy(model(Xtr[i:i + 10000]), Ytr[i:i + 10000], reduction="sum").item()
               for i in range(0, len(Xtr), 10000)) / len(Xtr)


def grads(model, idx):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
    return [l.weight.grad.detach().clone() for l in model.layers]


def full_grad(model):
    acc = None
    for i in range(0, len(Xtr), 5000):
        model.zero_grad(set_to_none=True)
        (F.cross_entropy(model(Xtr[i:i + 5000]), Ytr[i:i + 5000], reduction="sum") / len(Xtr)).backward()
        g = [l.weight.grad.detach().clone() for l in model.layers]
        acc = g if acc is None else [a + b for a, b in zip(acc, g)]
    return acc


GAMMAS = [0.01, 0.2, 1.0, 5.0, 20.0]
METHODS = [f"snowflake γ={g:g}" for g in GAMMAS] + ["normalized GD (M=I)", "muon", "newton-muon (γ=0.2)"]
CS = torch.logspace(-3.5, 0.5, 25).tolist()
from optim_compare import MLP
results = {}
for step, sd in snaps.items():
    model = MLP(Xtr.size(1)).cuda(); model.load_state_dict(sd)
    with torch.no_grad():                                           # exact M per layer, current weights
        Ms, cur = [], Xtr
        for i, l in enumerate(model.layers):
            Ms.append(norm_second_moment(cur))
            cur = F.relu(cur @ l.weight.T) if i < 2 else None
    Minv = {g: [damped_inverse(M, g) for M in Ms] for g in GAMMAS}
    L0 = full_loss(model)
    W0 = [l.weight.detach().clone() for l in model.layers]
    gen = torch.Generator(device="cuda").manual_seed(step)
    dl = {(mode, meth): torch.zeros(len(CS)) for mode in ["rms", "spectral"] for meth in METHODS}
    Gfull = full_grad(model)
    Gb_all = []
    for b in range(args.batches):
        idx = torch.randint(0, len(Xtr), (BATCH,), device="cuda", generator=gen)
        Gs = grads(model, idx); Gb_all.append(Gs)
        dirs = {}
        for g in GAMMAS: dirs[f"snowflake γ={g:g}"] = [G @ Mi for G, Mi in zip(Gs, Minv[g])]
        dirs["normalized GD (M=I)"] = Gs
        dirs["muon"] = [msgn(G) for G in Gs]
        dirs["newton-muon (γ=0.2)"] = [msgn(G @ Mi) for G, Mi in zip(Gs, Minv[0.2])]
        with torch.no_grad():
            for meth, D in dirs.items():
                for mode in ["rms", "spectral"]:
                    size = [((d @ M) * d).sum().sqrt() if mode == "rms" else torch.linalg.matrix_norm(d, 2)
                            for d, M in zip(D, Ms)]
                    unit = [d / s * (d.size(0) / d.size(1)) ** 0.5 for d, s in zip(D, size)]
                    for j, c in enumerate(CS):
                        for l, w0, u in zip(model.layers, W0, unit): l.weight.copy_(w0 - c * u)
                        dl[(mode, meth)][j] += (full_loss(model) - L0) / args.batches
                    for l, w0 in zip(model.layers, W0): l.weight.copy_(w0)
    # gradient SNR per eigendirection of M (layer 1)
    ev, V = torch.linalg.eigh(Ms[0]); ev, V = ev.flip(0), V.flip(1)
    sig = (Gfull[0] @ V).pow(2).sum(0)
    noise = torch.stack([((Gb[0] - Gfull[0]) @ V).pow(2).sum(0) for Gb in Gb_all]).mean(0)
    results[step] = dict(L0=L0, dl={f"{k[0]}|{k[1]}": v.tolist() for k, v in dl.items()},
                         ev=ev.cpu().tolist(), snr=(sig / noise).cpu().tolist())
    print(f"[{args.dataset}] step {step}: full train loss {L0:.4f}", flush=True)
    for mode in ["rms", "spectral"]:
        print(f"   matched {mode:8s} best Δloss (own best c):  " + "  ".join(
            f"{m}: {dl[(mode, m)].min():+.4f} @c={CS[int(dl[(mode, m)].argmin())]:.3g}" for m in METHODS), flush=True)

json.dump(dict(cfg=cfg, cs=CS, methods=METHODS, results=results), open(f"{OUT}/summary.json", "w"))

# plots
cols = dict(zip(METHODS, [plt.cm.viridis(x) for x in [0.0, 0.25, 0.5, 0.7, 0.85]] + ["0.5", "C1", "C3"]))
fig, axes = plt.subplots(len(results), 3, figsize=(19, 4.6 * len(results)), squeeze=False)
for r, (step, R) in enumerate(results.items()):
    for c_, mode in enumerate(["rms", "spectral"]):
        a = axes[r, c_]
        for m in METHODS:
            y = torch.tensor(R["dl"][f"{mode}|{m}"])
            a.plot(CS, y, color=cols[m], lw=2 if m in ("muon", "snowflake γ=5") else 1.3,
                   ls="--" if "muon" in m else "-", label=f"{m} (best {y.min():+.4f})")
        a.set_xscale("log"); a.axhline(0, color="k", lw=.6)
        lo = min(min(R["dl"][f"{mode}|{m}"]) for m in METHODS)
        a.set_ylim(lo * 1.15, -lo * 0.6)
        a.set_xlabel(f"step size c  (each layer's {mode} size = c·√(n/m))")
        a.set_ylabel("change in full training loss"); a.grid(alpha=.3, which="both"); a.legend(fontsize=7)
        a.set_title(f"step {step} (loss {R['L0']:.3f}): one step, sizes matched by {mode.upper()}")
    a = axes[r, 2]
    ev, snr = torch.tensor(R["ev"]), torch.tensor(R["snr"])
    a.loglog(range(1, len(ev) + 1), snr, ".", ms=3, color="C3", label="gradient SNR along eigvector k (batch 128)")
    a.axhline(1, color="C3", ls=":", lw=1)
    a2 = a.twinx(); a2.loglog(range(1, len(ev) + 1), ev.clamp_min(1e-12), color="C0", label="eigenvalue of M")
    a.set_xlabel("eigen-direction k of layer-1 M (sorted by eigenvalue)"); a.set_ylabel("signal / noise", color="C3")
    a2.set_ylabel("eigenvalue", color="C0"); a.grid(alpha=.3, which="both")
    a.set_title("layer 1: gradient SNR vs input-variance direction"); a.legend(loc="lower left", fontsize=7)
fig.suptitle(f"{args.dataset}: best one-step loss decrease of each update direction (exact M, batch gradient, avg of {args.batches} batches)")
fig.tight_layout(); fig.savefig(f"{OUT}/one_step_direction_test.png", dpi=110); plt.close(fig)
print("saved", OUT)
