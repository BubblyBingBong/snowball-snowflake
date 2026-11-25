#!/usr/bin/env bash
# Rerun every experiment, figure and table of the report (about 3 hours on one RTX 5090). Run from the repo root.
set -euo pipefail
python prepare_data.py
for ds in mnist cifar10; do
  python -m experiments.sweep --dataset $ds
  python -m experiments.snowflake_diag --dataset $ds
  python -m experiments.snowflake_oracle --dataset $ds
done
python -m experiments.final_runs
python -m experiments.effective_scaling
python -m plots.loss_curves
python -m plots.snowflake_diagnosis
python -m analysis.condition_number
python -m analysis.tables
