# Reproducing the report

Requires Python 3.10+ and one CUDA GPU.

```bash
pip install -r requirements.txt
```

## Option 1: tables and figures from the saved results (under a minute)

`results/` already contains every run summary used in the report.

```bash
python make_tables.py        # prints the numbers in every table and the diagnosis numbers quoted in the text
python paper_plot.py         # results/paper/loss_curves.pdf        (loss-curve figure)
python paper_diag_plot.py    # results/paper/snowflake_diagnosis.pdf (diagnosis figure)
python prepare_data.py && python condition_number.py   # condition numbers quoted in Results (needs the data)
```

## Option 2: rerun everything from scratch (about 3 hours on one RTX 5090)

```bash
./reproduce.sh
```

This downloads the data, runs every hyperparameter sweep, the extended damping sweep, the 3-seed final runs and
the Snowflake diagnostics, then regenerates both figures and prints all tables. Sweep runs are cached in
`results/**/runs/`, so an interrupted run resumes where it stopped. Step times depend on the GPU and on other
load, so the ms/step column will differ; losses and accuracies should match to the reported precision.

## Data

`prepare_data.py` downloads MNIST with torchvision and CIFAR-10 from the Hugging Face copy
(`uoft-cs/cifar10`). Use this script rather than another CIFAR-10 source: the validation split is a fixed random
subset of the training order, and torchvision's CIFAR-10 orders the training images differently.

## Where each result comes from

| Report item | Produced by |
|---|---|
| Main results table, loss-curve figure | `paper_runs.py`, `paper_plot.py` (configs from `ext_damping.py` and the sweeps) |
| Hyperparameter table | `optim_compare.py`, `input_aware.py`, `ext_damping.py` |
| Damping table | `ext_damping.py` |
| Diagnosis figure, oracle table | `mnist_muon.py`, `snowflake_oracle.py`, `snowflake_diag.py`, `paper_diag_plot.py` |
| Condition numbers in Results | `condition_number.py` |
