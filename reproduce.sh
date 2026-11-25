#!/usr/bin/env bash
# Reproduce every experiment, figure and table of the report (about 3 hours on one RTX 5090).
set -euo pipefail
python prepare_data.py                                   # MNIST + CIFAR-10 into data/
for ds in mnist cifar10; do
  python optim_compare.py   --dataset $ds                # AdamW / Muon / Newton-Muon / Snowflake sweeps
  python input_aware.py     --dataset $ds --methods iamuon   # Snowball sweep
  python snowflake_diag.py  --dataset $ds                # effective scaling, estimate error
  python snowflake_oracle.py --dataset $ds               # exact-M and M = I variants
done
python ext_damping.py                                    # extended damping grids -> final selections
python paper_runs.py                                     # 3-seed final runs (main results table)
python mnist_muon.py                                     # Muon effective-scaling ratio (diagnosis panel a)
python paper_plot.py                                     # loss-curve figure
python paper_diag_plot.py                                # diagnosis figure
python condition_number.py                               # condition numbers quoted in Results
python make_tables.py                                    # numbers for every table
