# Mini-Course on Machine Learning for Dynamic Economic Models (2026)

<p align="center">
  <img src="screens/duke_econ.png" alt="Duke Economics" height="90">
</p>

This is the teaching repository for the mini-course **Machine Learning for Dynamic Economic Models**, taught at the **Duke University Department of Economics**, in **September 2026**.

**Instructor:** [Yucheng Yang](https://sites.google.com/site/yangyucheng1993/home) — University of Zurich; ERID Visitor, Duke University · <yucheng.yang@uzh.ch>

Lecture slides, code, and readings are posted here as the course approaches.

## Schedule

The lectures take place over lunch, **12:00–1:15 PM**:

| Lecture | Date | Room |
|---|---|---|
| 1. Deep Learning for Solving Heterogeneous Agents Models | Tuesday, September 8 | SS105 |
| 2. Structural Reinforcement Learning for Macroeconomics | Wednesday, September 9 | SS113 |
| 3. Deep Learning for Continuous Time Models and Structural Estimation | Friday, September 11 | SS113 |

- **[Reading List (PDF)](Reading_List.pdf)** — suggested preparation plus core and background readings

---

## The three lectures

The course covers three machine-learning methods for solving heterogeneous agent models with aggregate shocks, moving from discrete time to continuous time and from a distribution-based to a price-based state space.

| # | Lecture | Slides | Method | Core reading | Code |
|---|---------|:---:|--------|--------------|------|
| 1 | **Deep Learning for Solving Heterogeneous Agents Models** | [PDF](Lectures/Lecture1_slides_DeepHAM.pdf) | Use neural networks to parameterize high-dimensional value and policy functions in heterogeneous agent models, with the cross-sectional distribution represented by *learned generalized moments*; trained along simulated paths. | [Han, Yang & E (2026)](Readings/DeepHAM_paper.pdf), *Quantitative Economics* | [Tutorial 1: DeepHAM on Colab](Tutorials/Tutorial1) |
| 2 | **Structural Reinforcement Learning for Macroeconomics** | [PDF](Lectures/Lecture2_slides_SRL.pdf) | Replace the distribution with low-dimensional *prices* as state variables; agents learn equilibrium price dynamics from simulated paths and optimize via structural policy gradient. | [Yang, Wang, Schaab & Moll (2025)](Readings/SRL_paper.pdf) | [SRL tutorials](Tutorials/SRL_tutorial_code.md) *(to be updated)* |
| 3 | **Deep Learning for Continuous Time Models and Structural Estimation** | [PDF](Lectures/Lecture3_slides_Continuous_Time_Structural_Estimation.pdf) | Search and matching with two-sided heterogeneity in continuous time: general equilibrium as a high-dimensional PDE with the distribution as a state variable, solved globally by deep learning and estimated via SMM. | [Payne, Rebei & Yang (2026)](Readings/DeepSAM_paper.pdf), *conditionally accepted, Econometrica* | *Coming soon* |

Materials: [`Lectures/`](Lectures) (slides) · [`Tutorials/`](Tutorials) (code walkthroughs) · [`Readings/`](Readings) (papers).

---

## Before the course

The lectures will be easier to follow if you have looked at the two suggested items in the [Reading List](Reading_List.pdf):

1. **Heterogeneous-agent models** — Dirk Krueger, [*An Introduction to Macroeconomics with Household Heterogeneity*](https://www.uni-bielefeld.de/fakultaeten/wirtschaftswissenschaften/einrichtungen/bigsem/profiles/economics/winter-term-2025-2026/HeteroBookLatex2025.pdf) (lecture notes), Chapter 6.
2. **Coding** — Python notebooks on solving simple Brock–Mirman models with neural networks: [deterministic](https://github.com/sischei/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models/blob/main/lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_01_Brock_Mirman_1972_DEQN.ipynb) and [stochastic](https://github.com/sischei/Deep_Learning_for_Solving_And_Estimating_Dynamic_Economic_Models/blob/main/lectures/lecture_03_deep_equilibrium_nets/code/lecture_03_02_Brock_Mirman_Uncertainty_DEQN.ipynb).

New to Python? See the [Python refresher](https://github.com/yangycpku/ML_Macro_Finance_Summer2026/tree/main/python_refresher).

If you plan to run the code, a [Google Colab](https://colab.research.google.com/) account with GPU access is the easiest setup — the tutorial notebooks open in Colab directly and need no local installation.

---

## Related course

A longer, five-day treatment of this material and of machine learning for macro-finance more broadly: the [PKU–Zurich PhD Summer School on Machine Learning for Macroeconomics and Finance](https://github.com/yangycpku/ML_Macro_Finance_Summer2026) (Beijing, July 2026).

---

## License

Teaching materials in this repository are released under the [Creative Commons CC0 1.0 Universal](LICENSE) license, except where individual files state otherwise (e.g. third-party code retained under its original license).
