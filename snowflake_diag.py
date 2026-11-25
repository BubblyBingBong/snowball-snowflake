"""Snowflake diagnostics, best Snowflake config from results/optim_compare/<dataset>/summary.json.

(1) Last training step: distribution over inputs of the effective scaling  s(x) = ||dW x|| / ||x||  per layer
    (dW = Snowflake update without weight decay, recorded inside the optimizer), on the training batch the
    update was computed from AND on 5000 held-out val inputs ("future" inputs). RMS marked.
(2) Every 50 steps: the optimizer's running estimate M_est of E[x x^T/||x||^2] vs the true M over the full
    training set with the current weights:
      - relative Frobenius error ||M_est - M_true||_F / ||M_true||_F
      - RMS of s(x) for this step's update, as predicted by M_est vs actual over the full training set
      - eigenvalue spectra at the last step

Usage: .venv/bin/python snowflake_diag.py --dataset mnist|cifar10
"""
import argparse, json, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optim_compare import load, train, MLP

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
ap.add_argument("--every", type=int, default=50)
args = ap.parse_args()
OUT = f"results/snowflake_diag/{args.dataset}"
os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(f"results/optim_compare/{args.dataset}/summary.json"))["snowflake"]["best"]
data = load(args.dataset)
(Xtr, Ytr), (Xva, Yva), _ = data
print("snowflake best config:", cfg)


@torch.no_grad()
def layer_inputs(model, X, chunk=5000):
    """List (per layer) of input matrices for all rows of X with the model's current weights."""
    outs = [[] for _ in model.layers]
    for i in range(0, len(X), chunk):
        model(X[i:i + chunk])
        for l, z in enumerate(model.inputs):
            outs[l].append(z)
    return [torch.cat(o) for o in outs]


@torch.no_grad()
def full_second_moments(model, X, chunk=2500):
    """Per-layer E[u u^T] over all rows of X (current weights), accumulated chunk by chunk to save memory."""
    acc = None
    for i in range(0, len(X), chunk):
        model(X[i:i + chunk])
        U = [z / z.norm(dim=1, keepdim=True).clamp_min(1e-12) for z in model.inputs]
        part = [(u.T @ u).double() for u in U]
        acc = part if acc is None else [a + b for a, b in zip(acc, part)]
    return [a / len(X) for a in acc]


@torch.no_grad()
def second_moment(Z):
    U = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (U.T @ U / len(U)).double()


def rms_under(dW, M):          # sqrt(E||dW u||^2) = sqrt(tr(dW M dW^T))
    dW = dW.double()
    return ((dW @ M) * dW).sum().clamp_min(0).sqrt().item()


log = dict(step=[], relerr=[[], [], []], rms_pred=[[], [], []], rms_pred_damped=[[], [], []], rms_true=[[], [], []],
           eps=[[], [], []], prior_weight=[])
last = {}


def cb(step, total, model, opt):
    if not (step % args.every == 0 or step == total):
        return
    params = opt.param_groups[0]["params"]
    Ztr_batch = [z.clone() for z in model.inputs]                # inputs of the batch this update came from
    Mfull = full_second_moments(model, Xtr)                      # full training set, current weights
    log["step"].append(step)
    log["prior_weight"].append(opt.beta ** (opt.t // opt.k))     # weight left on the I/m prior
    for l, p in enumerate(params):
        st = opt.state[p]
        m = p.size(1)
        Mest, Mtrue = st["K"].double(), Mfull[l]
        Mdamped = Mest + opt.gamma * Mest.trace() / m * torch.eye(m, device=Mest.device, dtype=Mest.dtype)
        dW = st["update"]
        log["relerr"][l].append(((Mest - Mtrue).norm() / Mtrue.norm()).item())
        log["rms_pred"][l].append(rms_under(dW, Mest) / st["eps"])
        log["rms_pred_damped"][l].append(rms_under(dW, Mdamped) / st["eps"])   # = 1 by construction
        log["rms_true"][l].append(rms_under(dW, Mtrue) / st["eps"])
        log["eps"][l].append(st["eps"])
    if step == total:
        Zva = layer_inputs(model, Xva)
        for l, p in enumerate(params):
            st = opt.state[p]
            dW = st["update"]
            s = lambda Z: ((Z @ dW.T).norm(dim=1) / Z.norm(dim=1).clamp_min(1e-12)) / st["eps"]
            last[l] = dict(s_batch=s(Ztr_batch[l]).cpu(), s_val=s(Zva[l]).cpu(),
                           sigma1=torch.linalg.matrix_norm(dW, 2).item() / st["eps"],
                           shape=tuple(p.shape),
                           ev_est=torch.linalg.eigvalsh(st["K"].double()).flip(0).cpu(),
                           ev_true=torch.linalg.eigvalsh(Mfull[l]).flip(0).cpu())


train("snowflake", cfg, data, dict(mnist=5, cifar10=10)[args.dataset], callback=cb, record=True)

# ---------------- (1) histogram of s(x) at the last step
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for l, ax in enumerate(axes):
    d = last[l]
    sb, sv = d["s_batch"], d["s_val"]
    hi = max(sb.max(), sv.max(), d["sigma1"]) * 1.05
    bins = torch.linspace(0, float(hi), 60)
    ax.hist(sv.numpy(), bins=bins.numpy(), density=True, alpha=0.45, color="C0", label=f"val inputs (n={len(sv)})")
    ax.hist(sb.numpy(), bins=bins.numpy(), density=True, alpha=0.6, color="C1", histtype="step", lw=2,
            label=f"train batch (n={len(sb)})")
    rms_b, rms_v = sb.pow(2).mean().sqrt().item(), sv.pow(2).mean().sqrt().item()
    ax.axvline(rms_b, color="C1", ls="--", lw=2, label=f"RMS train batch = {rms_b:.3f}")
    ax.axvline(rms_v, color="C0", ls="--", lw=2, label=f"RMS val = {rms_v:.3f}")
    ax.axvline(1.0, color="k", ls=":", lw=1.5, label="budget ε (=1)")
    ax.axvline(d["sigma1"], color="C3", ls="-.", lw=1.5, label=f"spectral norm σ₁ = {d['sigma1']:.3f}")
    ax.set_title(f"layer {l+1} {d['shape']}:  max/RMS  batch {sb.max()/rms_b:.2f}, val {sv.max()/rms_v:.2f}")
    ax.set_xlabel("effective scaling  ‖ΔW x‖/‖x‖   ÷ ε"); ax.set_ylabel("density"); ax.legend(fontsize=7.5)
    ax.grid(alpha=.3)
fig.suptitle(f"{args.dataset} Snowflake ({', '.join(f'{k}={v:g}' for k, v in cfg.items())}): effective scaling per input at the LAST step")
fig.tight_layout(); fig.savefig(f"{OUT}/last_step_effective_scaling.png", dpi=130); plt.close(fig)

# ---------------- (2) estimated vs true second moment
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
for l in range(3):
    a = axes[l, 0]
    a.plot(log["step"], log["relerr"][l], color="C3")
    a.set_yscale("log"); a.set_ylabel(f"layer {l+1} {last[l]['shape']}\n‖M_est − M_true‖_F / ‖M_true‖_F")
    a.grid(alpha=.3, which="both")
    a2 = a.twinx(); a2.plot(log["step"], log["prior_weight"], color="gray", ls=":", label="weight left on I/m prior")
    a2.set_ylim(0, 1); a2.set_ylabel("prior weight", color="gray")
    if l == 0: a.set_title("estimate error vs truth (full train set, current weights)"); a2.legend(loc="upper right", fontsize=8)

    a = axes[l, 1]
    a.plot(log["step"], log["rms_pred_damped"][l], color="k", ls=":", label="what the optimizer constrains (damped M_est) = ε")
    a.plot(log["step"], log["rms_pred"][l], color="C1", label="predicted by M_est (no damping)")
    a.plot(log["step"], log["rms_true"][l], color="C0", label="actual over full train set")
    a.set_ylabel("RMS of ‖ΔW x‖/‖x‖  ÷ ε"); a.grid(alpha=.3); a.set_ylim(0, None)
    if l == 0: a.set_title("RMS effective scaling of each update: predicted vs actual"); a.legend(fontsize=8)

    a = axes[l, 2]
    a.loglog(range(1, len(last[l]["ev_true"]) + 1), last[l]["ev_true"].clamp_min(1e-12), label="true M (full train set)")
    a.loglog(range(1, len(last[l]["ev_est"]) + 1), last[l]["ev_est"].clamp_min(1e-12), label="running estimate M_est")
    m = len(last[l]["ev_est"])
    a.axhline(cfg["gamma"] / m, color="k", ls=":", label=f"damping γ/m = {cfg['gamma']/m:.2e}")
    a.set_ylabel("eigenvalue"); a.grid(alpha=.3, which="both")
    if l == 0: a.set_title("eigenvalues at the last step"); a.legend(fontsize=8)
for a in axes[2]: a.set_xlabel("step" if a is not axes[2, 2] else "eigenvalue index")
fig.suptitle(f"{args.dataset} Snowflake: running second-moment estimate vs true second moment")
fig.tight_layout(); fig.savefig(f"{OUT}/second_moment_estimate_vs_true.png", dpi=110); plt.close(fig)

summary = {f"layer{l+1}": dict(shape=last[l]["shape"], sigma1_over_eps=last[l]["sigma1"],
                               rms_batch=last[l]["s_batch"].pow(2).mean().sqrt().item(),
                               rms_val=last[l]["s_val"].pow(2).mean().sqrt().item(),
                               max_batch=last[l]["s_batch"].max().item(), max_val=last[l]["s_val"].max().item(),
                               relerr_final=log["relerr"][l][-1],
                               rms_pred_mean=sum(log["rms_pred"][l]) / len(log["step"]),
                               rms_true_mean=sum(log["rms_true"][l]) / len(log["step"])) for l in range(3)}
json.dump(dict(cfg=cfg, layers=summary, log=log), open(f"{OUT}/summary.json", "w"), indent=1)
for k, v in summary.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
print("saved", OUT)
