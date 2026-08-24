<div align="center">

# Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics

Yucheng Yang, Chiyuan Wang, Andreas Schaab, Benjamin Moll

[![arXiv](https://img.shields.io/badge/arXiv-2512.18892-b31b1b.svg)](https://arxiv.org/abs/2512.18892)
[![SSRN](https://img.shields.io/badge/SSRN-5988734-133a6f.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5988734)
[![PDF](https://img.shields.io/badge/PDF-8A2BE2)](https://benjaminmoll.com/SRL/)

Link to toturial code repository: https://github.com/StructuralRL/SRL

</div>

This repo presents a pedagogical [JAX](https://github.com/jax-ml/jax) implementation of
policy-gradient methods for heterogeneous-agent macro models in general
equilibrium, including with aggregate risk.

SRL goes with the paper [*Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics*](https://benjaminmoll.com/SRL/) by [Yucheng Yang](https://sites.google.com/site/yangyucheng1993/home), [Chiyuan Wang](https://wangchiyuan.com/), [Andreas Schaab](http://andreasschaab.com/), and [Benjamin Moll](https://benjaminmoll.com). We try to make the repo and our codes as accessible as possible. Our hope is that you will be able to clone the repo, read the notebooks top to bottom, and easily follow along with both the economics and the method. Note this is the *tutorial repo* for SRL, not the final replication codes.

## The tutorials

Start with the tutorials. They start with the simplest, canonical household problem and build up to our full implementation of a heterogeneous agent model with aggregate risk by adding one step at a time. Read them in order from notebook 0. [TUTORIAL.md](TUTORIAL.md) is the full syllabus.

| # | Notebook | Open in Colab | Colab G4 runtime | Free Colab T4 runtime |
|---|----------|:-------------:|:--------:|:-------------:|
| 0 | [Household problem, in NumPy](tutorials_new/00_household_numpy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/00_household_numpy.ipynb) | ~1 min | ~3 min |
| 1 | [The same problem, in JAX](tutorials_new/01_household_jax.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/01_household_jax.ipynb) | ~6 s | ~15 s |
| 2 | [The same problem, by policy gradient](tutorials_new/02_household_policy_gradient.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/02_household_policy_gradient.ipynb) | ~30 s | ~1 min |
| 3 | [The household with moving prices](tutorials_new/03_household_moving_prices.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/03_household_moving_prices.ipynb) | ~45 s | ~1.5 min |
| 4 | [Huggett without aggregate risk](tutorials_new/04_ge_huggett.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/04_ge_huggett.ipynb) | ~4.5 min | ~8 min |
| 5 | [Huggett with aggregate risk](tutorials_new/05_ge_huggett_agg_risk.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/05_ge_huggett_agg_risk.ipynb) | ~35 s | ~1.5 min |
| 6 | [Huggett with Epstein-Zin preferences](tutorials_new/06_ge_huggett_epstein_zin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/06_ge_huggett_epstein_zin.ipynb) | ~5 min | ~10 min |

Each notebook runs top to bottom on a Colab GPU (click the badge) or locally on a CPU (but more slowly). On Colab the first cell clones the repo and installs the package, so there's no token and no setup notebook.

## Install (local)

```sh
pip install -e .
```

Python ≥ 3.11. JAX is pinned (`jax==0.10.0`) so results reproduce exactly across
machines.

## Repo structure

- **`srl/`** is the package: The policy-gradient solver (`SPGSolver`), a
  value-function-iteration baseline (`VFISolver`), and the discretization and
  plotting helpers (`srl.utils`).
- **`tutorials_new/`**: the notebooks, the shared `calibration.py`, and the SSJ reference data.
- **`TUTORIAL.md`** is the syllabus.

## License & citation

MIT, see [LICENSE](LICENSE). If you use SRL in your work, please cite our paper:

```bibtex
@misc{SRL,
      title={Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics},
      author={Yucheng Yang and Chiyuan Wang and Andreas Schaab and Benjamin Moll},
      year={2025},
      eprint={2512.18892},
      archivePrefix={arXiv},
      primaryClass={econ.TH},
      url={https://arxiv.org/abs/2512.18892},
}
```
