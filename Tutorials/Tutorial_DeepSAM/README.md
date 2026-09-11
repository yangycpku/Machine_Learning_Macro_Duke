# Tutorial 3: DeepSAM on Google Colab

Tutorial for **Lecture 3, Deep Learning for Continuous Time Models and Structural
Estimation**, of the Duke Mini-Course on Machine Learning for Dynamic Economic Models
(September 8–11, 2026).

The model is the labour search-and-matching economy of Section 3 of *Deep Learning for
Search and Matching Models* (Payne, Rebei & Yang): **two-sided heterogeneity** (worker types
$x$, firm types $y$), **aggregate shocks**, and **distributional feedback**, the
distribution of existing matches changing the value of forming new ones. Its state is an
aggregate shock $z$ together with the entire cross-sectional distribution $g$ of matches, 55
dimensions here. DeepSAM learns the match surplus $S(x, y, z, g)$ as a neural network that
takes $g$ directly as an input, trained on the residual of the master equation at states
drawn from simulating the model itself.

---

## The notebooks

Click a badge to open the notebook in Google Colab. No installation is needed: the first
cell clones this repository into the Colab runtime and moves into
`Tutorials/Tutorial_DeepSAM`, and Colab already ships PyTorch, NumPy, SciPy, matplotlib and
OmegaConf. Sign in with a Google account and pick a **GPU runtime** (`Runtime -> Change
runtime type`); the timings below are for an A100.

| # | Notebook | Open in Colab | What it does | Role | Runtime (`smoke`) |
|---|---|:---:|---|---|---|
| 01 | [`01_DeepSAM_Model_Loss_Training.ipynb`](notebooks/01_DeepSAM_Model_Loss_Training.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial_DeepSAM/notebooks/01_DeepSAM_Model_Loss_Training.ipynb) | The model and the master equation; then the **three elements** of the method in code: the surplus network, the master-equation residual term by term, and the two-phase sampler; then training (continued from the checkpoint vs. from scratch) and the residual map across type space | in-class walkthrough | ~3 min |
| 02 | [`02_DeepSAM_COVID_Exercise.ipynb`](notebooks/02_DeepSAM_COVID_Exercise.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial_DeepSAM/notebooks/02_DeepSAM_COVID_Exercise.ipynb) | Write the law of motion of the match distribution yourself (acceptance, marginals, meetings, KFE drift, Euler step, jumps), then reproduce Section 3: the COVID calibration, the recovery with and without re-sorting, and the mechanism | homework (8 blanks) | ~2 min once filled |
| 03 | [`03_DeepSAM_COVID_Solutions.ipynb`](notebooks/03_DeepSAM_COVID_Solutions.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial_DeepSAM/notebooks/03_DeepSAM_COVID_Solutions.ipynb) | Notebook 02 with every blank filled in | solutions | ~2 min |

Each blank in notebook 02 is a `TODO(n)` placeholder: cells that only define a function
run, and the first *call* raises a `NotImplementedError` naming the TODO to fill next. Every
exercise is followed by a check cell that compares the hand-written code with the library
implementation in `src/`, so you know when a blank is right.

## Run modes

Every notebook opens with

```python
RUN_MODE = "smoke"     # one of: "smoke", "teaching", "production"
```

and a cell that maps it onto the handful of numbers that drive wall-clock time.

| | `smoke` | `teaching` | `production` |
|---|---|---|---|
| ergodic-pool paths × horizon (01) | 32 × 500 | 64 × 1000 | 256 × 5000 |
| training steps, each of two runs (01) | 500 | 5,000 | 20,000 |
| pre-COVID ergodic paths × years (02, 03) | 50 × 10 | 100 × 20 | 200 × 30 |
| recovery paths averaged (02, 03) | 20 | 60 | 200 |
| wall clock on an A100, notebook 01 / 02 | ~3 / ~2 min | ~15 / ~3 min | ~1 h / ~6 min |

`production` matches the replication settings. Note what is *not* on this list: training the
surplus network to convergence. The full pipeline behind the shipped checkpoint is a
homotopy initialisation, a long main training phase run to a loss threshold, and then 8
further rounds of 100,000 gradient steps with the simulated dataset rebuilt between rounds,
several hours on an A100. That is why the notebooks load the checkpoint and why notebook 01
quantifies the gap a short run leaves rather than trying to close it.

## What ships here

```
config/config.yaml      calibration, network size, training settings
src/train_nn.py         the method: steady states, the surplus network, the master-equation
                        residual (S_pde_oper), the samplers, and the training loops
src/env.py              the economic environment: grids, production, matching, shocks
src/plotting.py         residual maps and the Section 3 figures
src/calibration_plot.py the pre-COVID ergodic distribution and the COVID calibration figure
src/covid_shock_plot.py the recovery simulations
checkpoints/            the converged surplus networks (baseline and symmetric)
```

`solve_steady_state()` writes its `.npy` files and an `agg_shock_*` folder into the project
root at run time; they are ignored by git.

## Running locally instead of on Colab

Clone the repository, install PyTorch (CUDA optional but much faster), NumPy, SciPy,
matplotlib, pandas and `omegaconf`, and open the notebooks from inside
`Tutorials/Tutorial_DeepSAM/notebooks` (the first cell steps up to the project root).
Verified on the Colab A100 runtime of September 2026 (Python 3.13, PyTorch 2.11).
