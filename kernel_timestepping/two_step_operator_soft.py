"""Step 2 of the two-step method, soft constraint: fit the operator on a frozen state.

With the state frozen at alpha_bar, the dynamic residual (theta_stepper.py) is linear in b. Its boundary rows do not
contain b; at the collocation nodes

    e_n = d_n - A_n b,   d_n = (K alpha_bar_{n+1} - K alpha_bar_n)_col,
                         A_n = dt [theta Psi(s_{n+1}) + (1 - theta) Psi(s_n)],

so for a set of active atoms (columns of A)

    J(b) = lam_dyn sum_n ||e_n||^2 + lam_beta sum_m beta_m^T K_m beta_m

has the closed-form minimiser b = (lam_dyn A^T A + lam_beta blockdiag(K_m))^-1 lam_dyn A^T d.

`OperatorRegression.cv` scores lam_beta by cross-validation over contiguous blocks of time steps (RMS of e on the
held-out steps); atom_selection.py uses it to choose the atoms. The hard-constraint Step 2 is in
two_step_operator_hard.py.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .cross_validation import rms
from .kmd_library import AtomLibrary


@dataclass(frozen=True)
class OperatorRegression:
    """The linear regression for b on a frozen state: e_n = d_n - A_n b at the collocation nodes (module docstring)."""
    library: AtomLibrary
    A: np.ndarray           # (N_steps, N_C, n_coef)
    d: np.ndarray           # (N_steps, N_C)
    penalty: np.ndarray     # (n_coef, n_coef) blockdiag(K_m)
    lam_dyn: float = 1.0

    @classmethod
    def from_state(cls, basis, library, alpha, dt, theta=0.5, lam_dyn=1.0):
        psi = np.asarray(library.design(basis.jets(jnp.asarray(alpha))))        # (N_T, N_C, n_coef)
        u = np.asarray(basis.values(alpha))[:, basis.collocation]
        return cls(library, dt * (theta * psi[1:] + (1 - theta) * psi[:-1]), u[1:] - u[:-1], library.penalty(),
                   lam_dyn)

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
        """e_n at the collocation nodes for the time steps `steps`: (len(steps), N_C)."""
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
