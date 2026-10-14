"""Space-time collocation grid, observations at its nodes, and cross-validation folds.

The state is represented column by column: time columns t_n = t0 + n dt (uniform) and spatial nodes x_k shared by
every column. Two kinds of boundary:

    fixed values (period=None)   x_0 and x_{N_X - 1} are boundary nodes whose values do not change in time; the PDE
                                 is collocated at the interior nodes x_1..x_{N_X-2}.
    periodic (period=P)          the nodes lie in one period [x_0, x_0 + P); there are no boundary nodes and the PDE
                                 is collocated at every node. Use a P-periodic state kernel (jaxkernels
                                 PeriodicKernel / PeriodicMaternKernel).

`collocation` and `boundary` hold the node indices of each kind; the rest of the package reads them from here.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from jaxkernels.objectives import kfold_indices


def blended_chebyshev_nodes(n, lo=0.0, hi=1.0, blend=0.5):
    """n increasing nodes on [lo, hi], end points included: blend * Chebyshev-Lobatto + (1 - blend) * uniform."""
    k = np.arange(n)
    reference = blend * (-np.cos(np.pi * k / (n - 1))) + (1 - blend) * (-1 + 2 * k / (n - 1))    # on [-1, 1]
    return lo + (hi - lo) * (reference + 1) / 2


def periodic_nodes(n, period, x0=0.0):
    """n uniform nodes x0 + period * k / n, k = 0..n-1: one period, the end point x0 + period left out."""
    return x0 + period * np.arange(n) / n


@dataclass(frozen=True)
class SpaceTimeGrid:
    """Spatial nodes x (N_X,), increasing, and N_T time columns t_n = t0 + n dt; period P for a periodic domain
    (module docstring), None for fixed boundary values at x[0] and x[-1]."""
    x: np.ndarray
    dt: float
    n_t: int
    t0: float = 0.0
    period: Optional[float] = None

    def __post_init__(self):
        x = np.asarray(self.x, dtype=float)
        if x.ndim != 1 or len(x) < 3 or np.any(np.diff(x) <= 0):
            raise ValueError("x must be a 1-D increasing array with at least three nodes")
        if self.dt <= 0 or self.n_t < 2:
            raise ValueError("need dt > 0 and at least two time columns")
        if self.period is not None and not x[-1] - x[0] < self.period:
            raise ValueError("periodic nodes must lie within one period (leave out the end point x[0] + period)")
        object.__setattr__(self, "x", x)

    @property
    def periodic(self):
        return self.period is not None

    @property
    def collocation(self):
        """Indices of the nodes where the PDE is collocated: the interior nodes, or every node if periodic."""
        return np.arange(self.n_x) if self.periodic else np.arange(1, self.n_x - 1)

    @property
    def boundary(self):
        """Indices of the boundary nodes, whose values do not change in time: [0, N_X - 1], or none if periodic."""
        return np.array([], dtype=int) if self.periodic else np.array([0, self.n_x - 1])

    @property
    def t(self):
        return self.t0 + self.dt * np.arange(self.n_t)

    @property
    def n_x(self):
        return len(self.x)

    @property
    def x_int(self):
        """The collocation nodes x[collocation] (the interior nodes unless periodic)."""
        return self.x[self.collocation]

    @property
    def n_steps(self):
        return self.n_t - 1


@dataclass(frozen=True)
class NodalObservations:
    """Observations y_i = u(t_{n_i}, x_{k_i}) at grid nodes: time indices n, space indices k, values y."""
    n: np.ndarray
    k: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        n, k, y = np.asarray(self.n, dtype=int), np.asarray(self.k, dtype=int), np.asarray(self.y, dtype=float)
        if not (n.shape == k.shape == y.shape and n.ndim == 1):
            raise ValueError("n, k and y must be 1-D arrays of the same length")
        for name, value in (("n", n), ("k", k), ("y", y)):
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.y)

    @classmethod
    def from_points(cls, grid, tx, y, atol=1e-8):
        """Observations given at coordinates tx (n_obs, 2) = (t, x), which must be grid nodes (within atol)."""
        tx = np.asarray(tx, dtype=float)
        n = np.abs(tx[:, :1] - grid.t[None, :]).argmin(1)
        k = np.abs(tx[:, 1:] - grid.x[None, :]).argmin(1)
        off = max(np.abs(tx[:, 0] - grid.t[n]).max(), np.abs(tx[:, 1] - grid.x[k]).max())
        if off > atol:
            raise ValueError(f"observation coordinates are up to {off:.2e} away from the nearest grid node")
        return cls(n, k, y)

    def take(self, rows):
        return NodalObservations(self.n[rows], self.k[rows], self.y[rows])

    def without(self, rows):
        return self.take(np.setdiff1d(np.arange(len(self)), rows))

    def is_boundary(self, grid):
        return np.isin(self.k, grid.boundary)

    def is_interior(self, grid):
        """Not on the boundary and not in the initial column: the observations that cross-validation holds out."""
        return (self.n > 0) & ~self.is_boundary(grid)


def interior_folds(grid, obs, n_folds=5, seed=0):
    """Random partition of the interior observations into n_folds index arrays (rows of obs).

    Boundary and initial-column observations are never held out: they pin the boundary values and the initial
    condition in every fit."""
    interior = np.flatnonzero(obs.is_interior(grid))
    return [interior[fold] for fold in kfold_indices(len(interior), n_folds, seed=seed)]


def time_blocks(n_steps, n_blocks=5):
    """Partition of the time steps 0..n_steps-1 into n_blocks contiguous blocks."""
    return kfold_indices(n_steps, n_blocks)


def sample_nodal_observations(grid, u, n_interior, seed=0):
    """Observations of u(t, x) at every boundary node, every node of the initial column, and n_interior nodes drawn
    without replacement from the other nodes of the later columns (all nodes if periodic). u maps arrays (t, x) to
    values."""
    rng = np.random.default_rng(seed)
    n_all, k_all = np.meshgrid(np.arange(grid.n_t), np.arange(grid.n_x), indexing="ij")
    n_all, k_all = n_all.ravel(), k_all.ravel()
    fixed = (n_all == 0) | np.isin(k_all, grid.boundary)
    drawn = rng.choice(np.flatnonzero(~fixed), size=n_interior, replace=False)
    rows = np.concatenate([np.flatnonzero(fixed), np.sort(drawn)])
    n, k = n_all[rows], k_all[rows]
    return NodalObservations(n, k, np.asarray(u(grid.t[n], grid.x[k])))
