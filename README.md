# Charged Particle Simulator

**Numerical comparison of time-integration schemes for interacting charged particles**

## Overview

This project implements a 2D N-body simulation of charged particles interacting through:

* Coulomb forces
* Optional uniform electric field
* Optional uniform magnetic field
* Optional inelastic fusion on collision

The primary objective is **not visualization**, but rigorous comparison of numerical time-integration methods with respect to:

* Energy conservation
* Stability
* Long-time behavior
* Computational cost

Implemented integrators:

* Explicit Euler
* Velocity Verlet (symplectic)
* Classical RK4

---

## Project Structure

```
project/
│
├── src/
│   ├── simulator.py      # Core particle system + physics
│   ├── integrators.py    # Euler, Verlet, RK4 implementations
│   ├── experiments.py    # Generates CSV data
│   ├── metrics.py        # Energy drift metrics
│   └── animation.py      # Optional visualization
│
├── data/
│   └── *.csv             # Simulation outputs
│
├── notebooks/
│   └── analysis.ipynb    # Error + stability analysis
│
└── README.md
```

Separation of concerns:

* `experiments.py` generates datasets.
* The notebook performs analysis only.
* Core logic is isolated in `src/`.

---

## Mathematical Model

For particle ( i ):

$$
m_i \frac{d\mathbf{v}*i}{dt} = \sum*{j \ne i} \frac{k q_i q_j}{|\mathbf{r}\times{ij}|^3} \mathbf{r}*{ij}

\times q_i \left( \mathbf{E} + \mathbf{v}_i \times \mathbf{B} \right)
$$

Where:

* Coulomb interaction is pairwise.
* Magnetic force is perpendicular to velocity.
* Fusion (if enabled) conserves momentum but not energy.

---

## Numerical Methods

### Euler

* First order
* Fast
* Energy diverges rapidly
* Included as instability baseline

### Velocity Verlet

* Second order
* Symplectic
* Bounded energy oscillations
* Good long-term qualitative behavior

### RK4

* Fourth order
* Low local truncation error
* Not symplectic
* Exhibits secular energy drift

---

## Running Experiments

Generate datasets:

```bash
python src/experiments.py
```

This produces CSV files in `data/`.

Each CSV contains:

* Positions: `x_i`, `y_i`
* Velocities: `vx_i`, `vy_i`
* Energy diagnostics
* Alive flags (if fusion enabled)

---

## Analysis

Open:

```
notebooks/analysis.ipynb
```

The notebook:

* Computes relative energy drift
* Compares integrators across Δt
* Estimates convergence slopes
* Produces stability and runtime plots

---

## Metrics

Energy drift is defined as:

$$
\frac{|E(t) - E(0)|}{|E(0)|}
$$

Utility functions are implemented in:

```
src/metrics.py
```

---

## Key Findings

* Euler is unstable except at extremely small Δt.
* Verlet preserves qualitative dynamics even for long runs.
* RK4 achieves higher short-term accuracy but accumulates long-term drift.
* Symplectic structure matters more than local order for Hamiltonian systems.

---

## Requirements

All requirements can be found in `environment.yml`

---

## Notes

This project focuses on **numerical behavior**, not high-performance optimization.
For large N, complexity is $(O(N^2))$ due to pairwise force computation.