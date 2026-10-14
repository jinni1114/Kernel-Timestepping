"""One-step method, hard constraint: the state is the rollout of its first column, fitted together with the operator
coefficients, for a fixed set of atoms (from atom_selection.py).

Unknowns alpha_0 and b; every later column is alpha_n = Phi^n(alpha_0; b), so the dynamics hold exactly and the
lam_dyn term of the soft loss (one_step_soft.py) drops out. The remaining terms,

    lam_d sum_{(n,k) observed} (y_nk - (K alpha_n)_k)^2 + lam_alpha alpha_0^T K alpha_0
          + lam_beta sum_m beta_m^T K_m beta_m + lam_addot / dt^4 sum_n ||alpha_{n+1} - 2 alpha_n + alpha_{n-1}||_K^2,

form a residual vector minimised by trust-region Gauss-Newton, differentiating through the rollout. The library
stays frozen.

The state norm is taken on alpha_0 only, the one free state variable, not on every column as in the soft loss. The
rolled-out columns alpha_n, n >= 1, solve linear systems with a matrix close to K at every step; with an
ill-conditioned K their RKHS norm is huge even when their nodal values match the data (about 3e9 in whitened units
for the Burgers walkthrough, against 3.6 for the Step-1 state), so a penalty on them outweighs the data misfit and
biases the coefficients (u u_x: 0.49 instead of 1 at lam_alpha = 1e-6). The lam_addot term is also taken on
rolled-out columns and has the same problem; it is 0 by default.
"""
import jax.numpy as jnp
import numpy as np

from .jax_optimizers import least_squares_jax
from .one_step_soft import OneStepWeights
from .two_step_state_fit import second_difference


def fit_one_step_hard(stepper, library, grid, obs, alpha0_init, coef_init, weights=OneStepWeights(), max_nfev=100):
    """Minimise the hard one-step loss over (alpha_0, b) from (alpha0_init (N_X,), coef_init).

    Returns (alpha (N_T, N_X) = the rollout of the fitted alpha_0, coef, result)."""
    basis, n_x, n_steps = stepper.basis, grid.n_x, grid.n_steps
    K, chol, chol_inv = jnp.asarray(basis.K), jnp.asarray(basis.chol), jnp.asarray(basis.chol_inv)
    penalty_chol = jnp.asarray(np.linalg.cholesky(library.penalty() + 1e-10 * np.eye(library.n_coef)))
    D_tt = jnp.asarray(second_difference(grid.n_t))
    n_obs, k_obs, y_obs = jnp.asarray(obs.n), jnp.asarray(obs.k), jnp.asarray(obs.y)

    def trajectory(p):
        return stepper.rollout(p[:n_x] @ chol_inv, p[n_x:], n_steps)

    def residual(p):
        alpha = trajectory(p)
        u, w = alpha @ K, alpha @ chol
        return jnp.concatenate([np.sqrt(weights.data) * (u[n_obs, k_obs] - y_obs),
                                np.sqrt(weights.alpha) * p[:n_x],                    # w_0 = L^T alpha_0 only
                                np.sqrt(weights.beta) * (penalty_chol.T @ p[n_x:]),
                                np.sqrt(weights.addot) / grid.dt ** 2 * (D_tt @ w).ravel()])

    p0 = np.concatenate([np.asarray(basis.whiten(alpha0_init)), np.asarray(coef_init)])
    result = least_squares_jax(residual, p0, max_nfev=max_nfev)
    return np.asarray(trajectory(jnp.asarray(result.x))), result.x[n_x:], result
