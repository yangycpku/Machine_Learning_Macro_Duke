# Tutorial 1: DeepHAM on Google Colab

Tutorial for **Lecture 1, Deep Learning for Solving Heterogeneous Agents Models**, of the Duke
Mini-Course on Machine Learning for Dynamic Economic Models (September 8–11, 2026).

DeepHAM solves heterogeneous-agent models with aggregate shocks by (i) representing the
wealth distribution with a small number of **generalized moments**, basis functions learned
jointly with the solution rather than chosen by hand, and (ii) improving the policy by
differentiating through a simulation of the economy. The notebooks below build that up on
the Krusell–Smith (1998) model.

Paper: Han, Yang & E (2026), *DeepHAM: A global solution method for heterogeneous agent
models with aggregate shocks*, **Quantitative Economics** 17(2), 297–341.
Reference implementation: <https://github.com/frankhan91/DeepHAM>.

---

## The notebooks

Click a badge to open the notebook in Google Colab. No installation is needed: the first
cell clones this repository into the Colab runtime and moves into `Tutorials/Tutorial1/src`,
and Colab already ships TensorFlow, NumPy, SciPy and matplotlib. Sign in with a Google
account; a GPU is optional (see *Run modes*).

| # | Notebook | Open in Colab | Topic | Role | Runtime (`smoke`) |
|---|---|:---:|---|---|---|
| 01 | [`01_DeepHAM_KS_FixedMoment.ipynb`](src/01_DeepHAM_KS_FixedMoment.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial1/src/01_DeepHAM_KS_FixedMoment.ipynb) | Krusell–Smith solved with the cross-sectional **mean** as the only distribution statistic (`n_fm=1, n_gm=0`) | in-class walkthrough | ~3.5 min on CPU |
| 02 | [`02_DeepHAM_KS_GeneralizedMoment.ipynb`](src/02_DeepHAM_KS_GeneralizedMoment.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial1/src/02_DeepHAM_KS_GeneralizedMoment.ipynb) | The same run with a **learned** generalized moment (`n_fm=0, n_gm=1`): one config change | in-class walkthrough | ~3.5 min on CPU |
| 03 | [`03_DeepHAM_Policy_Exercise.ipynb`](src/03_DeepHAM_Policy_Exercise.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial1/src/03_DeepHAM_Policy_Exercise.ipynb) | Write the policy objective yourself: prices, budget constraint, `stop_gradient`, the unrolled utility sum | exercise (5 blanks) | as 01 once filled |
| 04 | [`04_DeepHAM_Policy_Solutions.ipynb`](src/04_DeepHAM_Policy_Solutions.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial1/src/04_DeepHAM_Policy_Solutions.ipynb) | Solutions to 03, with commentary on the three details that matter | solution | as 01 |
| 05 | [`05_DeepHAM_Visualize.ipynb`](src/05_DeepHAM_Visualize.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yangycpku/Machine_Learning_Macro_Duke/blob/main/Tutorials/Tutorial1/src/05_DeepHAM_Visualize.ipynb) | What the generalized moment learned; reproduces Figures 3 and 7 of the paper | in-class walkthrough | ~30 s on CPU |

Notebook 05 needs no training: it loads the solved model shipped in `data/simul_results`.
Start there if you want the punchline before the machinery.

## Run modes

Every training notebook opens with

```python
RUN_MODE = "smoke"   # one of: "smoke", "teaching", "production"
```

and a cell that maps it onto the handful of numbers that drive wall-clock time.

| | `smoke` | `teaching` | `production` |
|---|---|---|---|
| policy gradient steps | 100 | 1,500 | 10,000 |
| unroll horizon | 60 | 150 | 150 |
| simulated paths | 64 | 192 | 384 |
| value-net epochs | 10 | 60 | 200 |
| runtime, A100 GPU | 2.2 min | 15.6 min | ~90 min |
| runtime, Colab free CPU | ~3.5 min | | |
| mean capital reached | ~11 | ~32 | ~39 (the KS level) |

`smoke` exercises every code path (dataset construction, value fitting, the policy loop,
the periodic value refresh, saving) but is far too short to converge: it lands around
$K \approx 11$ against the Krusell–Smith level of $K \approx 39$, while `teaching` reaches
about 32. `production` is the setting behind the published results (the reference run in
`data/simul_results` took 5,278 s on an A100).

Use `smoke` in class. For a `teaching` run, switch the Colab runtime to a GPU first
(`Runtime -> Change runtime type -> T4 GPU`, free tier) and expect roughly the A100 time
scaled up; on the CPU runtime it is too slow for a session.

Runs are written to `data/simul_results/KS/game_nn_n50_<exp>_<RUN_MODE>` inside the cloned
repository, so a quick smoke run can never overwrite a long one, nor the reference solutions
below. On Colab that folder lives on the runtime's disk and disappears when the runtime is
recycled: each training notebook ends with an optional cell that downloads a zip of the run.

## What ships in `data/`

| | |
|---|---|
| `KS_policy_value_NS.mat` | the Krusell–Smith benchmark policy, as b-splines: the comparison target throughout |
| `simul_results/KS/game_nn_n50_1fm1` | solved model, 1 fixed moment (notebook 01) |
| `simul_results/KS/game_nn_n50_1gm3` | solved model, 1 generalized moment: the run behind Figure 3 of the paper (notebook 05) |

The `matlab/` folder holds the scripts that generate the benchmark `.mat` file; you do not
need to run them.

`src/` also contains the JFV and Dávila model code from the full DeepHAM repository
(`train_JFV.py`, `simulation_Davila.py`, ...). Those are here so you can read them, but the
`.mat` inputs they need are not shipped; get them from the
[reference repository](https://github.com/frankhan91/DeepHAM) if you want to run them.

## Running locally instead of on Colab

Clone the repository, install TensorFlow 2.x, NumPy, SciPy, matplotlib and tqdm, and open
the notebooks from inside `Tutorials/Tutorial1/src` (the modules are imported from the
working directory and the data is read from `../data`). No GPU is required at `smoke` and
`teaching` budgets.

Verified on the Colab runtime of September 2026 (Python 3.13, TensorFlow 2.20, Keras 3.13).
Note that `value.py` and `policy.py` deliberately leave `prepare_state`, `value_fn` and
`policy_fn` **undecorated**: nesting `@tf.function` methods breaks from TF 2.20 onward, where
`self` inside a traced method becomes a `TfMethodTarget` that cannot resolve another
`tf.function` attribute. They are only ever called from inside `loss`/`train_step`, which are
traced, so nothing is lost.
