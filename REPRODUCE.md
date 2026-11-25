# Reproducing the report

Requires Python 3.10+ and one CUDA GPU. Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

## Option 1: figures and tables from the saved results (under a minute)

`results/` contains the outputs of every experiment in the report.

```bash
python -m analysis.tables              # numbers in every table, and the diagnosis numbers quoted in the text
python -m plots.loss_curves            # results/figures/loss_curves.pdf
python -m plots.snowflake_diagnosis    # results/figures/snowflake_diagnosis.pdf
python prepare_data.py && python -m analysis.condition_number   # condition numbers quoted in Results
```

## Option 2: rerun everything (about 3 hours on one RTX 5090)

```bash
./reproduce.sh
```

This downloads the data, runs every sweep, the 3-seed final runs and the diagnostics, then regenerates both figures
and prints all tables. Individual runs are cached in `results/*/runs/`, so an interrupted run resumes where it
stopped. Step times depend on the GPU and its load; losses and accuracies should match to the reported precision.

## Data

`prepare_data.py` downloads MNIST with torchvision and CIFAR-10 from the Hugging Face copy (`uoft-cs/cifar10`).
Use this script rather than another CIFAR-10 source: the validation split is a fixed random subset of the training
order, and torchvision's CIFAR-10 orders the training images differently.

## Where each result comes from

| Report item | Produced by |
|---|---|
| Main results table, loss-curve figure | `experiments/sweep.py`, `experiments/final_runs.py`, `plots/loss_curves.py` |
| Hyperparameter and damping tables | `experiments/sweep.py` |
| Diagnosis figure and oracle table | `experiments/effective_scaling.py`, `experiments/snowflake_oracle.py`, `plots/snowflake_diagnosis.py` |
| Estimate-accuracy and effective-scaling numbers | `experiments/snowflake_diag.py`, `experiments/effective_scaling.py` |
| Condition numbers in Results | `analysis/condition_number.py` |
