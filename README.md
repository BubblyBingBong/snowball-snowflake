# Snowball and Snowflake: input-aware versions of Muon

Code and results for the CS182 report *Input-Aware Optimizers*. Muon takes the steepest-descent step under the
spectral norm, a worst case over all input directions. The two optimizers here measure update size on the inputs a
layer actually receives instead:

- **Snowball** whitens the layer inputs (multiplying by `C^{-1/2}` turns their distribution into a unit ball) and
  takes a Muon step in those coordinates: `dW = -eta * polar(G C^{-1/2}) C^{-1/2}`.
  In the code it is called `iam` / `iamuon` (its earlier name, Input-Aware Muon). Its tuned damping is very heavy
  (gamma = 500), where it is close to Muon; as gamma -> infinity it is exactly Muon.
- **Snowflake** bounds the root-mean-square change in the layer's outputs over the input distribution:
  `dW = -eps * G M^{-1} / sqrt(tr(G M^{-1} G^T))`, with `M = E[x x^T / ||x||^2]`.

Baselines: AdamW, Muon, Newton-Muon (Du & Su, arXiv 2604.01472). All experiments use a bias-free MLP
(d_in -> 512 -> 512 -> 10) on MNIST (5 epochs) and CIFAR-10 (10 epochs), batch 128.

## Files

| Script | What it does | Output |
|---|---|---|
| `optim_compare.py` | Model, data, Muon / Newton-Muon / Snowflake optimizers; hyperparameter sweeps for AdamW, Muon, Newton-Muon, Snowflake | `results/optim_compare/` |
| `input_aware.py` | Snowball (`iamuon`) and other input-aware variants; sweep for Snowball | `results/input_aware/` |
| `ext_damping.py` | Extends the damping grids for Snowball (up to 500) and Snowflake (up to 100); its selections are used for the final runs | `results/paper/extended_damping.json` |
| `paper_runs.py` | Final 3-seed runs of every method at its selected configuration (Table 2 of the report) | `results/paper/{mnist,cifar10}.pt`, `summary.json` |
| `paper_plot.py` | Loss-curve figure (Figure 1) | `results/paper/loss_curves.pdf` |
| `mnist_muon.py` | Effective scaling / spectral norm of Muon updates on MNIST (diagnosis, hypothesis 1) | `results/final_histories.pt` |
| `snowflake_diag.py` | Snowflake effective-scaling distribution and second-moment estimate error | `results/snowflake_diag/` |
| `snowflake_oracle.py` | Snowflake with an exact (oracle) second moment and with `M = I` (diagnosis, hypothesis 2) | `results/snowflake_oracle/` |
| `snowflake_direction.py` | One-step direction test and gradient SNR along input eigendirections | `results/snowflake_direction/` |
| `prepare_data.py` | Downloads MNIST and CIFAR-10 into `data/` | `data/` |
| `make_tables.py` | Prints the numbers in every table | stdout |
| `condition_number.py` | Condition numbers of the damped metric quoted in Results | stdout |
| `paper_diag_plot.py` | Diagnosis figure (Appendix C) from the saved results above | `results/paper/snowflake_diagnosis.pdf` |

Sweep caches (`results/**/runs/`) are not included; the scripts recreate them.

## Reproducing

See [REPRODUCE.md](REPRODUCE.md): tables and figures from the saved results take under a minute, and
`./reproduce.sh` reruns every experiment from scratch.

All runs used one NVIDIA RTX 5090 in fp32 (TF32 disabled), PyTorch 2.15.
