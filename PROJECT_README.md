# SOS Basis Pruning and QRL Result Organization Project

This document reorganizes the project based on the actual files in the current `Thesis` directory. The project trains a reinforcement-learning model to prune SOS Gram matrix bases and generates, verifies, and archives the corresponding `Q_RL` results for real polynomial examples.

## Project Goals

For an even-degree polynomial `f(x)`, the SOS representation can be written as:

```text
f(x) = z(x)^T Q z(x),  Q >= 0
```

Here `z(x)` is the full monomial basis and `Q` is a positive semidefinite Gram matrix. This project uses Double DQN + Prioritized Experience Replay to train a pruning policy that minimizes the retained basis while preserving SDP feasibility, yielding:

```text
f(x) = z_RL(x)^T Q_RL z_RL(x)
```

The project also contains organized results for 20 appendix examples, `C1` through `C20`; each sample directory stores the corresponding polynomial, benchmark row, and `Q_RL` files.

## Current Directory Structure

```text
Thesis/
├── PROJECT_README.md
├── train.xlsx
├── solve_sos_qrl.py
├── src/
│   ├── __init__.py
│   ├── generator.py
│   ├── env.py
│   ├── model.py
│   ├── agent.py
│   ├── buffer.py
│   └── util.py
├── all/
│   ├── bench_sdp.py
│   ├── basis_red_boxes.csv
│   ├── res.csv
│   ├── result_real.csv
│   ├── result_real.md
│   ├── qrl_results/
│   └── qrl_results_scs_check/
├── C1/ ... C20/
└── 2n6d/, 2n8d/, 2n10d/, 3n4d/, 3n6d/, 3n8d/,
    4n4d/, 4n6d/, 5n4d/, 6n4d/, 7n4d/, 8n2d/
```

## Core Code

| File | Purpose |
|---|---|
| `src/generator.py` | Builds the SOS monomial basis, the full polynomial coefficient space, and the `Q -> P` coefficient map, then generates training/test data. |
| `src/env.py` | `SOSPruningEnvGT` reinforcement-learning environment; the state is the coefficient vector and current mask, and actions remove a basis term or STOP. |
| `src/model.py` | `TwoStreamQNet`, which processes the polynomial coefficient stream and mask stream separately and outputs Q-values for each pruning action and STOP. |
| `src/agent.py` | `DoubleDQNAgentPER`, including Double DQN, epsilon-greedy exploration, target network, and PER replay. |
| `src/buffer.py` | `SumTree` and `PrioritizedReplayBuffer`. |
| `src/util.py` | Utilities for random seeds, directory creation, training-curve plotting, and agent evaluation. |
| `solve_sos_qrl.py` | Reads polynomials and solves `Q_RL` under the reduced basis, writing CSV, NPZ, and JSON outputs. |
| `all/bench_sdp.py` | Runs RL, Newton Polytope, TSSOS, random pruning, and SDP timing for examples in `all/result_real.csv`. |

## Experiment Directories

Each `{n}n{d}d` directory corresponds to variable count `n` and total polynomial degree `d`; for example, `3n6d` means 3 variables and degree 6. Common files are:

| File/Directory | Purpose |
|---|---|
| `generate.py` | Generates `data/train.json` and `data/test.json`. |
| `train.py` | Trains the RL pruning model for the current configuration. |
| `eval.py` | Newton Polytope baseline evaluation. |
| `env_custom.py` | Custom environment parameters in some directories. |
| `model/` | Trained model weights, commonly named `train.pth`. |
| `picture/` | Training-curve images. |
| `result/` | Evaluation output directory. |

Existing experiment configurations:

```text
2n6d, 2n8d, 2n10d,
3n4d, 3n6d, 3n8d,
4n4d, 4n6d,
5n4d, 6n4d, 7n4d, 8n2d
```

## Runtime Environment

Use an existing Python/Conda environment. Project dependencies include:

| Package | Use |
|---|---|
| `torch` | Reinforcement-learning model and training. |
| `numpy` | Numerical computation. |
| `sympy` | Polynomial parsing and expansion. |
| `cvxpy` | SDP modeling and solving. |
| `scipy` | Linear programming for Newton Polytope. |
| `matplotlib` | Training-curve plotting. |
| `openpyxl` | Reading Excel datasets. |

`bench_sdp.py` uses MOSEK for timing by default; `solve_sos_qrl.py` uses CLARABEL by default and can switch to SCS when needed.

## Common Commands

Run from the project root:

```powershell
cd Thesis
```

Generate data for a configuration:

```powershell
cd 3n4d
python generate.py
```

Train a model for a configuration:

```powershell
cd 3n4d
python train.py
```

Run the Newton Polytope baseline:

```powershell
cd 3n4d
python eval.py
```

Solve `Q_RL` for real examples:

```powershell
cd Thesis
python solve_sos_qrl.py --csv all\result_real.csv --all --outdir all\qrl_results
```

Solve a single example:

```powershell
python solve_sos_qrl.py --csv all\result_real.csv --sample-id 33 --outdir all\qrl_results
```

Solve with the full basis:

```powershell
python solve_sos_qrl.py --expr "x1**2 + x2**2" --nvars 2 --degree 2 --full-basis
```

Run the comprehensive benchmark:

```powershell
cd all
python bench_sdp.py
```

## `all/` Result Description

| File/Directory | Content |
|---|---|
| `all/result_real.csv` | Real polynomial examples and basis sizes after full-basis, RL, NP, and TSSOS pruning. |
| `all/result_real.md` | Markdown version of `result_real.csv`. |
| `all/res.csv` | Comprehensive benchmark results, including SDP times and speedups for full-basis, RL, NP, TSSOS, and random pruning. |
| `all/qrl_results/` | `Q_RL` results for each example, including `*_Q_RL.csv`, `.npz`, and `.json`. |
| `all/qrl_results/summary.csv` | `Q_RL` solve summary: sample ID, configuration, basis size, minimum eigenvalue, rank, residual, and status. |
| `all/qrl_results_scs_check/` | Additional SCS check results; currently includes sample 30. |
| `all/basis_red_boxes.csv` | Auxiliary data related to basis reduction. |

The current `all/res.csv` has 20 examples. Statistics from the existing results:

| Metric | Average |
|---|---:|
| Full basis size | 24.2 |
| After RL pruning | 9.4 |
| After NP pruning | 13.3 |
| After TSSOS pruning | 13.7 |
| RandomPrunedSize | 21.2 |

The current `all/qrl_results/summary.csv` has 20 `optimal` records, with maximum reconstruction residual about `2.68e-10`.

## `C1` Through `C20` Archive Directories

`C1` through `C20` are standalone directories organized from appendix examples. Each directory contains:

| File/Directory | Content |
|---|---|
| `qrl_results/` | `Q_RL` CSV, NPZ, and JSON for the example. |
| `res.csv` | The single benchmark-result row for the example from `all/res.csv`. |
| `result_real.csv` | The single polynomial record for the example from `all/result_real.csv`. |
| `polynomial.txt` | The polynomial text and sample ID for the example. |
| `qrl_summary.csv` | The single summary row for the example from `all/qrl_results/summary.csv`. |
| `manifest.json` | Example number, sample ID, QRL file count, and extra-check flag. |

Example mapping:

| C ID | sample id |
|---|---:|
| C1 | 8 |
| C2 | 11 |
| C3 | 15 |
| C4 | 16 |
| C5 | 21 |
| C6 | 22 |
| C7 | 30 |
| C8 | 35 |
| C9 | 54 |
| C10 | 55 |
| C11 | 57 |
| C12 | 58 |
| C13 | 59 |
| C14 | 60 |
| C15 | 65 |
| C16 | 101 |
| C17 | 3 |
| C18 | 9 |
| C19 | 10 |
| C20 | 33 |

`C7` also contains `qrl_results_scs_check/`, corresponding to additional SCS check results for sample 30.

## `Q_RL` Output Format

`solve_sos_qrl.py` writes three files for each example:

| File | Content |
|---|---|
| `sample_<id>_<config>_Q_RL.csv` | Numeric matrix for the Gram matrix `Q_RL`. |
| `sample_<id>_<config>.npz` | Compressed data package containing `Q_RL`, full basis, RL basis, active indices, rank, residual, and related data. |
| `sample_<id>_<config>.json` | Readable metadata containing the polynomial, basis monomials, solve status, minimum eigenvalue, rank, and residual. |

Example:

```text
all/qrl_results/sample_33_4n6d_Q_RL.csv
all/qrl_results/sample_33_4n6d.npz
all/qrl_results/sample_33_4n6d.json
```

## Model Configuration

`solve_sos_qrl.py` and `all/bench_sdp.py` use the same model-width configuration:

```python
MODEL_CONFIG = {
    "2n6d":  {"base_dim": 512,  "embed_dim": 16},
    "2n8d":  {"base_dim": 512,  "embed_dim": 16},
    "2n10d": {"base_dim": 1024, "embed_dim": 32},
    "3n4d":  {"base_dim": 128,  "embed_dim": 8},
    "3n6d":  {"base_dim": 256,  "embed_dim": 8},
    "3n8d":  {"base_dim": 1024, "embed_dim": 32},
    "4n4d":  {"base_dim": 256,  "embed_dim": 8},
    "4n6d":  {"base_dim": 1024, "embed_dim": 32},
    "5n4d":  {"base_dim": 256,  "embed_dim": 8},
    "6n4d":  {"base_dim": 1024, "embed_dim": 32},
    "7n4d":  {"base_dim": 4096, "embed_dim": 256},
    "8n2d":  {"base_dim": 128,  "embed_dim": 8},
}
```

Model weights are read from the corresponding directory by default:

```text
<config>/model/train.pth
```

For example `3n6d/model/train.pth`.

## Notes

- Scripts in experiment directories usually import `src/` via `sys.path.append("..")`; run `generate.py`, `train.py`, and `eval.py` inside the corresponding experiment directory.
- `bench_sdp.py` uses `all/result_real.csv` and assumes a trained `model/train.pth` exists in the corresponding configuration directory.
- If MOSEK is unavailable, MOSEK timing in `bench_sdp.py` will fail; use `solve_sos_qrl.py --solver CLARABEL` or `--solver SCS` for single-example validation instead.
- Some source comments in the project previously had historical encoding issues, but the core Python identifiers and current README use UTF-8.
