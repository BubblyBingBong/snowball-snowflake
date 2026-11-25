"""Is Snowflake held back by its second-moment estimate (H2)?  Three Snowflake variants, same update rule:

  est       normal Snowflake (EMA estimate of M, refreshed every 10 steps)        -- sweep from experiments/sweep.py
  oracle    exact M every step, one step ahead: layers are updated front to back; layer i's M is
            E[u u^T] (u = input/||input||) over the WHOLE training set, pushed through the already-updated
            layers 1..i-1, i.e. exactly the inputs layer i will see at the next step. Layer 1's M is fixed.
  identity  M = I/m forever (no input awareness) -> normalized gradient descent with momentum

Sweep: oracle lr x gamma (wd=0.1, the best wd of est Snowflake on both datasets), identity lr x wd.
Best config (lowest final val loss, seed 0) of each variant + Muon is then rerun with seeds 0,1,2 for test accuracy.

Usage: python -m experiments.snowflake_oracle --dataset mnist|cifar10  ->  results/diagnosis/snowflake_oracle/{dataset}.json
"""
import argparse, itertools, json, math, os, time
import torch, torch.nn.functional as F
from core import load, train, evaluate, MatrixOpt, damped_inverse, cfg_str


@torch.no_grad()
def norm_second_moment(Z, chunk=10000):
    m = Z.size(1)
    M = torch.zeros(m, m, device=Z.device)
    for i in range(0, len(Z), chunk):
        U = Z[i:i + chunk]
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        M += U.T @ U
    return M / len(Z)


class SnowflakeVariant(MatrixOpt):
    def __init__(self, model, mode, lr, wd, gamma, Xfull):
        super().__init__(model, "snowflake", lr, wd, gamma=gamma)
        self.mode, self.X = mode, Xfull
        p0 = self.param_groups[0]["params"][0]
        if mode == "oracle":                                   # layer 1 inputs never change: exact once
            self.state[p0]["K"] = norm_second_moment(Xfull)
            self.state[p0]["Kinv"] = damped_inverse(self.state[p0]["K"], gamma)

    @torch.no_grad()
    def step(self):
        if self.mode == "est":
            return super().step()
        self.t += 1
        g = self.param_groups[0]
        params = g["params"]
        cur = self.X
        for i, p in enumerate(params):
            st, G = self.state[p], p.grad
            n, m = p.shape
            if self.mode == "oracle" and i > 0:
                cur = F.relu(cur @ params[i - 1].T)            # params[i-1] was already updated this step
                st["K"] = norm_second_moment(cur)
                st["Kinv"] = damped_inverse(st["K"], self.gamma)
            buf = st["buf"].mul_(g["momentum"]).add_(G)
            u = G.add(buf, alpha=g["momentum"])                # Nesterov
            p.mul_(1 - g["lr"] * g["weight_decay"])
            GM = u @ st["Kinv"]
            eps = g["lr"] * math.sqrt(n / m)
            p.sub_(GM * (eps * (GM * u).sum().clamp_min(1e-30).rsqrt()))


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()
EPOCHS = dict(mnist=5, cifar10=10)[args.dataset]
OUT = "results/diagnosis/snowflake_oracle"
CACHE = f"results/diagnosis/runs/{args.dataset}"
os.makedirs(CACHE, exist_ok=True)
data = load(args.dataset)
Xtr = data[0][0]
prev = json.load(open(f"results/sweeps/{args.dataset}.json"))


def run(variant, cfg, seed=0, model_out=False):
    """variant in {est, oracle, identity, muon}. Returns (model or None, history)."""
    if variant == "muon":
        return train("muon", cfg, data, EPOCHS, seed=seed)
    mk = lambda model: SnowflakeVariant(model, variant, cfg["lr"], cfg["wd"], cfg.get("gamma", 0.0), Xtr)
    return train("snowflake", cfg, data, EPOCHS, seed=seed, make_opt=mk)


LRS = [0.01, 0.02, 0.05, 0.1, 0.2]
grids = dict(oracle=dict(lr=LRS, wd=[0.1], gamma=[0.01, 0.05, 0.2, 1.0, 5.0, 20.0]),
             identity=dict(lr=[0.0005, 0.001, 0.002, 0.005] + LRS + [0.5], wd=[0.1, 0.3]))
sweeps, best = {}, dict(est=prev["snowflake"]["best"], muon=prev["muon"]["best"])
sweeps["est"] = [(s["cfg"], s["val_loss"], s["val_acc"]) for s in prev["snowflake"]["sweep"]]
for variant, grid in grids.items():
    rows, t0 = [], time.time()
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(grid, vals))
        cache = f"{CACHE}/{variant}__{cfg_str(cfg).replace(', ', '_')}.pt"
        if os.path.exists(cache):
            h = torch.load(cache)
        else:
            _, h = run(variant, cfg); torch.save(h, cache)
        vl = h["val_loss"][-1] if math.isfinite(h["val_loss"][-1]) else float("inf")
        rows.append((cfg, vl, h["val_acc"][-1]))
        print(f"[{args.dataset}] {variant:9s} {cfg_str(cfg):34s} val_loss={vl:.4f} val_acc={h['val_acc'][-1]*100:.2f}%  "
              f"median step {torch.tensor(h['step_time']).median():.1f} ms", flush=True)
    rows.sort(key=lambda r: r[1])
    sweeps[variant], best[variant] = rows, rows[0][0]
    print(f"[{args.dataset}] BEST {variant}: {cfg_str(best[variant])}  ({time.time()-t0:.0f}s)", flush=True)

# multi-seed reruns of each variant's best config
VARIANTS = ["muon", "est", "oracle", "identity"]
final = {v: [] for v in VARIANTS}
for v in VARIANTS:
    for s in range(args.seeds):
        model, h = run(v, best[v], seed=s)
        tl, ta = evaluate(model, *data[2])
        final[v].append(dict(seed=s, val_loss=h["val_loss"][-1], val_acc=h["val_acc"][-1], test_loss=tl, test_acc=ta,
                             median_step_ms=torch.tensor(h["step_time"]).median().item()))
        print(f"[{args.dataset}] FINAL {v:9s} seed {s} {cfg_str(best[v]):34s} val_loss={h['val_loss'][-1]:.4f} "
              f"test_acc={ta*100:.2f}%", flush=True)

stat = lambda xs: (sum(xs) / len(xs), (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5)
summary = dict(best=best, sweeps={k: [dict(cfg=c, val_loss=a, val_acc=b) for c, a, b in r] for k, r in sweeps.items()},
               final=final)
print(f"\n[{args.dataset}] mean ± std over {args.seeds} seeds:")
for v in VARIANTS:
    (tam, tas), (vlm, vls) = stat([r["test_acc"] * 100 for r in final[v]]), stat([r["val_loss"] for r in final[v]])
    summary.setdefault("table", {})[v] = dict(cfg=best[v], test_acc_mean=tam, test_acc_std=tas, val_loss_mean=vlm, val_loss_std=vls)
    print(f"  {v:9s} {cfg_str(best[v]):34s} test acc {tam:.2f} ± {tas:.2f}%   val loss {vlm:.4f} ± {vls:.4f}")
json.dump(summary, open(f"{OUT}/{args.dataset}.json", "w"), indent=2)
