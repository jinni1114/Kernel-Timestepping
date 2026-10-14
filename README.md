# Kernel-Timestepping

Learns an evolution equation $u_t = P(u, u_x, u_{xx}, \dots)$ from sparse observations of one solution: a spatial
kernel state at every time column, a KMD operator library, and a theta-scheme time step tying them together.
The algorithm is in [`docs/algorithm.md`](docs/algorithm.md).

## Methods

| | soft constraint | hard constraint |
|---|---|---|
| **two-step** | `two_step_state_fit.py` → `two_step_operator_soft.py` | `two_step_state_fit.py` → `two_step_operator_hard.py` |
| **one-step** | `one_step_soft.py` | `one_step_hard.py` |

The atoms are selected once (`atom_selection.py`, on the two-step soft regression); the other three methods refit the
selected atoms, started from the two-step soft fit.

## Layout

```
kernel_timestepping/   the method (installed)
problems/              reference solvers for experiments and tests (not installed): Burgers (fixed boundary values);
                       advection-diffusion, KdV, Kuramoto-Sivashinsky (periodic, spectral ETDRK4)
experiments/           notebooks, one per experiment; burgers_walkthrough.ipynb goes through every stage;
                       advection_diffusion, kdv, kuramoto_sivashinsky run the same pipeline on periodic domains
archive/               original notebooks, kept frozen (KEqL Fig. 3B comparison; SINDy/4DVar)
docs/                  algorithm description
tests/                 pytest
```

## Boundaries

`SpaceTimeGrid(x, dt, n_t)` holds the values at `x[0]` and `x[-1]` fixed and collocates the PDE at the interior nodes.
`SpaceTimeGrid(kt.periodic_nodes(n, P), dt, n_t, period=P)` is periodic: the PDE is collocated at every node; use a
periodic state kernel (`jaxkernels.PeriodicKernel`) and, for smooth kernels, `state_lengthscale(..., max_cond=1e12)`
to keep $K$ invertible. Features go up to the derivative order of the basis (`max_order=4` for KS).

## Run

```bash
pip install -e ".[examples,test]"
pytest -q tests
```

```python
import kernel_timestepping as kt
from jaxkernels import GaussianRBFKernel

model = kt.KMDTimeStepper(GaussianRBFKernel(lengthscale=kt.state_lengthscale(grid, obs)), grid)
model.fit(obs, method="two_step_soft")      # or "two_step_hard", "one_step_soft", "one_step_hard"
print(model.summary())
u = model.rollout(u0_nodes, n_steps=50)
```
