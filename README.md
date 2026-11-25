# Snowball and Snowflake: input-aware versions of Muon

Code and results for the CS182 report *Input-Aware Muon Optimizers*. Muon takes the steepest-descent step under the
spectral norm, a worst case over all input directions. We propose two optimizers that instead measure update size
on the inputs a layer actually receives:

- **Snowball** whitens the layer inputs (multiplying by `C^{-1/2}` turns their distribution into a unit ball) and
  takes a Muon step in those coordinates: `dW = -eta * polar(G C^{-1/2}) C^{-1/2}`.
- **Snowflake** bounds the root-mean-square change in the layer's outputs over the input distribution:
  `dW = -eps * G M^{-1} / sqrt(tr(G M^{-1} G^T))`, with `M = E[x x^T / ||x||^2]`.

Both are compared with AdamW, Muon and Newton-Muon on a bias-free MLP (d_in -> 512 -> 512 -> 10) on MNIST and
CIFAR-10. To reproduce the results, see [REPRODUCE.md](REPRODUCE.md).

## Layout

```
core.py                  data loading, model, training loop, and all optimizers (Muon, Newton-Muon, Snowflake, Snowball)
prepare_data.py          downloads MNIST and CIFAR-10 into data/
experiments/
  sweep.py               hyperparameter sweeps for every method              -> results/sweeps/{dataset}.json
  final_runs.py          3 seeds at the selected configurations              -> results/final/
  effective_scaling.py   diagnosis, hypothesis 1 (do inputs align with the top singular direction?)
  snowflake_diag.py      diagnosis: accuracy of the running second-moment estimate
  snowflake_oracle.py    diagnosis, hypothesis 2: Snowflake with the exact second moment and with M = I
plots/                   loss_curves.py, snowflake_diagnosis.py               -> results/figures/
analysis/                tables.py (numbers in every table), condition_number.py
results/                 saved outputs of all of the above
```

Run every script from the repository root as a module, for example `python -m experiments.sweep --dataset mnist`.
