"""Condition number of the damped first-layer metric H = M + gamma * tr(M)/d_in * I for gamma = 5 and 500, with M
the exact second moment of the length-normalized training inputs (quoted in the Results section).
Usage: python -m analysis.condition_number
"""
import torch
from core import load

for ds in ["mnist", "cifar10"]:
    (X, _), _, _ = load(ds)
    U = X / X.norm(dim=1, keepdim=True)
    ev = torch.linalg.eigvalsh(U.T @ U / len(U))
    for g in [5, 500]:
        H = ev + g / len(ev)
        print(f"{ds:8s} gamma={g:<4d} cond(H) = {(H.max() / H.min()).item():.2f}")
