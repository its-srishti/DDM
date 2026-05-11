# Drift-Diffusion Model — From Scratch

A hand-coded implementation of the Drift-Diffusion Model (DDM) in Python, covering simulation, analytical likelihood, and MLE-based parameter estimation — built without relying on existing toolboxes like HDDM or PyDDM.

---

## What This Is

The DDM is a sequential sampling model for two-alternative forced-choice (2AFC) tasks. It models a decision as a noisy accumulation of evidence toward one of two boundaries. Four parameters govern the process:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Drift rate | `v` | Rate of evidence accumulation (signal strength) |
| Boundary separation | `a` | Distance between upper and lower decision boundaries |
| Starting point | `z` | Initial bias as a proportion of `a` (0.5 = unbiased) |
| Non-decision time | `t0` | Sensory + motor latency, in seconds |

---

## Repository Structure

```
DDM/
├── ddm_core.py               # Core library: simulation, likelihood, fitting
├── 01_simulate.ipynb         # Basic simulation and RT distribution plots
├── 02_simulate.ipynb         # Extended simulation across parameter regimes
├── 03_fit.ipynb              # MLE fitting pipeline and point-estimate tests
├── 04_recovery.ipynb         # Parameter recovery study
├── 05_validation.ipynb       # Likelihood surface scans and validation
└── 06_recovery_extended.ipynb # Recovery across biased starting points
```

---

## Core Functions (`ddm_core.py`)

### `simulate_ddm(v, a, z, t0, n_trials)`

Simulates trials using the Euler-Maruyama method. Returns an `(n_trials, 2)` array where column 0 is reaction time and column 1 is choice (1 = upper boundary, 0 = lower boundary).

```python
from ddm_core import simulate_ddm

data = simulate_ddm(v=0.3, a=1.2, z=0.5, t0=0.15, n_trials=300)
```

### `fit_ddm(data)`

Fits DDM parameters to observed data via Maximum Likelihood Estimation. Uses the Navarro-Fuss (2009) analytical likelihood with L-BFGS-B optimization and multiple random restarts.

```python
from ddm_core import fit_ddm

estimated = fit_ddm(data)
print(estimated)
# {'v': 0.312, 'a': 1.288, 'z': 0.502, 't0': 0.148, 'log_likelihood': ..., 'converged': True}
```

---

## Likelihood Implementation

The log-likelihood uses the **Navarro & Fuss (2009)** infinite-series solution to the first-passage time density. A single `pdf_upper` function is defined for the upper boundary; lower-boundary trials are handled via the **reflection principle**:

> A lower-boundary trial with parameters `(v, z)` has the same first-passage density as an upper-boundary trial evaluated at `(-v, 1-z)`.

```python
if choice == 1:
    density = pdf_upper(t, v, a, z)
else:
    density = pdf_upper(t, -v, a, 1 - z)
```

Key implementation details:
- Fourier series truncated at **K = 50** terms for convergence across extreme `z` values
- Girsanov exponent: `exp(+v·a·(1-z) − 0.5·v²·t)` — positive sign is critical
- Eigenfunction argument: `sin(k·π·(1-z))` — distance to upper boundary, not `z`
- Numerical guards: `np.clip(arg, -500, 500)`, density floor at `1e-10`

---

## Optimizer

`fit_ddm` uses **L-BFGS-B** with the following bounds:

```python
bounds = [
    (0.0, 5.0),           # v ≥ 0  (breaks the (v,z) / (-v,1-z) symmetry)
    (0.1, 5.0),           # a > 0
    (0.05, 0.95),         # z in open interval (0, 1)
    (0.01, min_rt * 0.9)  # t0 strictly less than fastest observed RT
]
```

The `v ≥ 0` bound is the key identifiability constraint: the DDM has a fundamental symmetry where `(v, z)` and `(-v, 1-z)` produce identical data. Bounding `v` to non-negative values confines the search to one mode, consistent with the convention in HDDM and PyDDM.

---

## Parameter Recovery

All tests at N = 300 trials unless noted. Errors are consistent with sampling noise at this sample size.

| True parameters | Estimated |
|----------------|-----------|
| v=0.3, a=1.2, z=0.5, t₀=0.15 | v=0.312, a=1.288, z=0.502, t₀=0.148 |
| v=0.3, a=1.2, z=0.7, t₀=0.15 | v=0.315, a=1.254, z=0.689, t₀=0.149 |
| v=0.3, a=1.2, z=0.3, t₀=0.15 | v=0.338, a=1.263, z=0.312, t₀=0.150 |
| v=0.5, a=1.2, z=0.3, t₀=0.15 | v=0.538, a=1.242, z=0.288, t₀=0.153 |
| v=0.7, a=1.2, z=0.7, t₀=0.15 | v=0.726, a=1.281, z=0.694, t₀=0.147 |
| v=0.3, a=1.2, z=0.7, t₀=0.15 (N=500) | v=0.316, a=1.288, z=0.697, t₀=0.148 |

Log-likelihood surface scans confirm smooth, unimodal surfaces peaking near the true value for all four parameters under both biased and unbiased conditions.

---

## Scope and Conventions

- Drift rate `v` is restricted to non-negative values throughout. In a 2AFC task, the upper boundary is defined as the correct/target response, so `v ≥ 0` means at or above chance performance.
- Three drift regimes are examined: weak (`v = 0.1`), moderate (`v = 0.3, 0.5`), and strong (`v = 0.8`).
- All simulations use fixed `a = 1.2`, `t0 = 0.15`; only `v` and `z` are varied across conditions.

---

## Requirements

```
numpy
scipy
matplotlib
jupyter
```

Install with:

```bash
pip install numpy scipy matplotlib jupyter
```

---

## Planned Extensions

- Full recovery study: N ∈ {50, 100, 200, 500}, 50 simulations per condition
- Bayesian estimation arm using PyMC
- MLE vs. Bayesian comparison as a function of N and parameter regime

---

## Reference

Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *Journal of Mathematical Psychology, 53*(4), 222–230.
