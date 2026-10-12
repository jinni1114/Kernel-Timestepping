"""Step 2 of the two-step method: fit the operator on a frozen state, and select its atoms.

Soft constraint. With the state frozen at alpha_bar, the dynamic residual (dynamics.py) is linear in b. Its boundary
rows do not contain b; at the interior nodes

    e_n = d_n - A_n b,   d_n = (K alpha_bar_{n+1} - K alpha_bar_n)_int,
                         A_n = dt [theta Psi(s_{n+1}) + (1 - theta) Psi(s_n)],

so for a set of active atoms (columns of A)

    J(b) = lam_dyn sum_n ||e_n||^2 + lam_beta sum_m beta_m^T K_m beta_m

has the closed-form minimiser b = (lam_dyn A^T A + lam_beta blockdiag(K_m))^-1 lam_dyn A^T d.

Selection. lam_beta is chosen by cross-validation over contiguous blocks of time steps (score: RMS of e on the
held-out steps). Atoms are selected by backward elimination: starting from all atoms, drop at each step the atom
whose removal gives the lowest CV score (lam_beta re-chosen for every candidate set), down to one prunable atom.
Among the nested sets on that path, the one-standard-error rule picks the smallest set whose CV score is within one
standard error of the best score on the path.

Hard constraint. `fit_operator_through_rollout` refits the coefficients of a fixed atom set so that the rollout of
the stepper from alpha_bar_0 matches the frozen state at every later column.
"""
from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .cv import rms
from .grid import time_blocks
from .library import AtomLibrary
from .optimize import least_squares_jax


@dataclass(frozen=True)
class OperatorRegression:
    """The linear regression for b on a frozen state: e_n = d_n - A_n b at the interior nodes (module docstring)."""
    library: AtomLibrary
    A: np.ndarray           # (N_steps, N_X - 2, n_coef)
    d: np.ndarray           # (N_steps, N_X - 2)
    penalty: np.ndarray     # (n_coef, n_coef) blockdiag(K_m)
    lam_dyn: float = 1.0

    @classmethod
    def from_state(cls, basis, library, alpha, dt, theta=0.5, lam_dyn=1.0):
        psi = np.asarray(library.design(basis.jets(jnp.asarray(alpha))))        # (N_T, N_X - 2, n_coef)
        u = np.asarray(basis.values(alpha))
        return cls(library, dt * (theta * psi[1:] + (1 - theta) * psi[:-1]), u[1:, 1:-1] - u[:-1, 1:-1],
                   library.penalty(), lam_dyn)

    @property
    def n_steps(self):
        return self.A.shape[0]

    def _normal_equations(self, cols, steps):
        A, d = self.A[steps][..., cols].reshape(-1, len(cols)), self.d[steps].ravel()
        return A.T @ A, A.T @ d

    def _solve(self, gram, rhs, cols, lam_beta):
        """(lam_dyn A^T A + lam_beta K_reg)^-1 lam_dyn A^T d; a relative ridge of 1e-12 keeps the solve defined
        when lam_beta is tiny."""
        H = self.lam_dyn * gram + lam_beta * self.penalty[np.ix_(cols, cols)]
        H = H + 1e-12 * np.trace(H) / len(cols) * np.eye(len(cols))
        return np.linalg.solve(H, self.lam_dyn * rhs)

    def fit(self, atoms, lam_beta, steps=None):
        """Minimiser of J(b) over the coefficients of `atoms` (stacked in that order), from the time steps `steps`
        (default: all)."""
        cols = self.library.columns(atoms)
        steps = np.arange(self.n_steps) if steps is None else np.asarray(steps)
        return self._solve(*self._normal_equations(cols, steps), cols, lam_beta)

    def residual(self, atoms, coef, steps=None):
        """e_n at the interior nodes for the time steps `steps`: (len(steps), N_X - 2)."""
        steps = np.arange(self.n_steps) if steps is None else np.asarray(steps)
        return self.d[steps] - self.A[steps][..., self.library.columns(atoms)] @ coef

    def cv(self, atoms, lam_grid, blocks):
        """Time-block CV for every lam_beta on lam_grid: fit without each block of steps, score e on it.

        Returns (RMS per block (n_lam, n_blocks), RMS pooled over the blocks (n_lam,))."""
        cols = self.library.columns(atoms)
        held_out = [[] for _ in lam_grid]
        for block in blocks:
            gram, rhs = self._normal_equations(cols, np.setdiff1d(np.arange(self.n_steps), block))
            A_block, d_block = self.A[block][..., cols], self.d[block]
            for i, lam_beta in enumerate(lam_grid):
                held_out[i].append((d_block - A_block @ self._solve(gram, rhs, cols, lam_beta)).ravel())
        return (np.array([[rms(r) for r in rows] for rows in held_out]),
                np.array([rms(np.concatenate(rows)) for rows in held_out]))


@dataclass(frozen=True)
class AtomSet:
    """An atom set scored by time-block CV, with lam_beta chosen on the grid."""
    atoms: tuple                # atom indices in the full library
    cv: float                   # pooled CV score at the chosen lam_beta
    se: float                   # standard error of the block scores at the chosen lam_beta
    lam_beta: float
    cv_curve: np.ndarray        # (n_lam,) pooled CV score along the lam_beta grid
    cv_blocks: np.ndarray       # (n_lam, n_blocks)
    dropped: Optional[int] = None    # the atom removed from this set at the next step of the path


def score_atom_set(regression, atoms, lam_grid, blocks):
    cv_blocks, cv_curve = regression.cv(atoms, lam_grid, blocks)
    best = int(np.argmin(cv_curve))
    return AtomSet(tuple(atoms), float(cv_curve[best]), float(cv_blocks[best].std(ddof=1) / np.sqrt(len(blocks))),
                   float(lam_grid[best]), cv_curve, cv_blocks)


def backward_elimination(regression, lam_grid, blocks):
    """Path of nested atom sets, from all atoms down to one prunable atom (non-prunable atoms always stay)."""
    library = regression.library
    current = score_atom_set(regression, range(len(library.atoms)), lam_grid, blocks)
    path = []
    while True:
        prunable = [m for m in current.atoms if library.atoms[m].prunable]
        if len(prunable) <= 1:
            path.append(current)
            return path
        candidates = [(score_atom_set(regression, [a for a in current.atoms if a != m], lam_grid, blocks), m)
                      for m in prunable]
        best, dropped = min(candidates, key=lambda c: (c[0].cv, c[1]))
        path.append(replace(current, dropped=dropped))
        current = best


def one_standard_error(path):
    """(i_min, i_sel, threshold): i_min has the lowest CV score on the path; i_sel is the last (smallest) set with
    cv <= threshold = cv_min + se_min."""
    i_min = int(np.argmin([step.cv for step in path]))
    threshold = path[i_min].cv + path[i_min].se
    i_sel = max(i for i, step in enumerate(path) if step.cv <= threshold)
    return i_min, i_sel, threshold


@dataclass(frozen=True)
class AtomSelection:
    """Result of select_atoms: the elimination path, the selected set and its fitted coefficients."""
    path: list
    i_min: int
    i_sel: int
    threshold: float
    lam_grid: np.ndarray
    coef: np.ndarray            # coefficients of the selected atoms, stacked in the order of `atoms`

    @property
    def selected(self):
        return self.path[self.i_sel]

    @property
    def atoms(self):
        return self.selected.atoms

    @property
    def lam_beta(self):
        return self.selected.lam_beta


def select_atoms(regression, lam_grid=None, n_blocks=5):
    """Backward elimination + one-standard-error rule, then the fit of the selected set on all time steps."""
    lam_grid = np.logspace(-12, 2, 15) if lam_grid is None else np.asarray(lam_grid, dtype=float)
    path = backward_elimination(regression, lam_grid, time_blocks(regression.n_steps, n_blocks))
    i_min, i_sel, threshold = one_standard_error(path)
    coef = regression.fit(path[i_sel].atoms, path[i_sel].lam_beta)
    return AtomSelection(path, i_min, i_sel, threshold, lam_grid, coef)


def fit_operator_through_rollout(stepper, library, alpha_ref, coef_init, lam_beta=1e-6, max_nfev=300):
    """Coefficients of `library` (the active atoms) fitted through the rollout, state frozen at alpha_ref:

        min_b sum_{n >= 1} ||K alpha_ref_n - K Phi^n(alpha_ref_0; b)||^2 + lam_beta sum_m beta_m^T K_m beta_m.

    Nonlinear in b through the rollout: trust-region Gauss-Newton with the exact Jacobian (forward mode through
    the Newton steps). `stepper` must use `library.evaluate` as its operator. Returns (coef, scipy result).
    """
    basis, n_steps = stepper.basis, alpha_ref.shape[0] - 1
    alpha0, target = jnp.asarray(alpha_ref[0]), basis.values(jnp.asarray(alpha_ref[1:]))
    penalty_chol = jnp.asarray(np.linalg.cholesky(library.penalty() + 1e-10 * np.eye(library.n_coef)))

    def residual(coef):
        rolled = basis.values(stepper.rollout(alpha0, coef, n_steps)[1:])
        return jnp.concatenate([(target - rolled).ravel(), jnp.sqrt(lam_beta) * (penalty_chol.T @ coef)])

    result = least_squares_jax(residual, coef_init, max_nfev=max_nfev)
    return result.x, result
