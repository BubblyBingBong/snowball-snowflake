"""Condition number of the damped first-layer metric H = M + gamma*tr(M)/d_in * I for gamma = 5 and 500
(Results, Q1 paragraph: 'falls from 145 at gamma = 5 to 2.4 at gamma = 500' on CIFAR-10). M is the exact second moment of the
length-normalized training inputs. Usage: python condition_number.py
"""
import torch
from optim_compare import load

for ds in ["mnist", "cifar10"]:
    (X, _), _, _ = load(ds)
    U = X / X.norm(dim=1, keepdim=True)
    ev = torch.linalg.eigvalsh(U.T @ U / len(U))
    d = len(ev)
    for g in [5, 500]:
        H = ev + g / d
        print(f"{ds:8s} gamma={g:<4d} cond(H) = {(H.max() / H.min()).item():.2f}")
