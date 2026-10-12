"""One-step methods: the state and the operator coefficients are fitted together, for a fixed set of atoms.

Soft constraint (`fit_joint_soft`): unknowns alpha_0..alpha_{N_T-1} and b,

    J(alpha, b) = lam_d     sum_{(n,k) observed} (y_nk - (K alpha_n)_k)^2
                + lam_dyn   sum_n ||e(alpha_{n+1}, alpha_n; b)||^2                  (all rows of e, dynamics.py)
                + lam_alpha sum_n alpha_n^T K alpha_n + lam_beta sum_m beta_m^T K_m beta_m
                + lam_addot / dt^4 sum_n ||alpha_{n+1} - 2 alpha_n + alpha_{n-1}||_K^2,

minimised by L-BFGS-B in the whitened state coordinates w_n = L^T alpha_n.

Hard constraint (`fit_joint_hard`): unknowns alpha_0 and b; every later column is the rollout
alpha_n = Phi^n(alpha_0; b), so the dynamics hold exactly and the lam_dyn term drops out. The remaining terms form a
residual vector that is minimised by trust-region Gauss-Newton, differentiating through the rollout.

The library (standardisation, inducing points, active atoms) stays frozen in both.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .optimize import lbfgs_jax, least_squares_jax
from .state_fit import second_difference


@dataclass(frozen=True)
class JointWeights:
    """Weights of the one-step loss (module docstring)."""
    data: float = 1.0       # lam_d
    dyn: float = 1.0        # lam_dyn (soft constraint only)
    alpha: float = 1e-6     # lam_alpha
    beta: float = 1e-8      # lam_beta
    addot: float = 0.0      # lam_addot


def fit_joint_soft(stepper, library, grid, obs, alpha_init, coef_init, weights=JointWeights(), maxiter=20000):
    """Minimise the soft one-step loss from (alpha_init (N_T, N_X), coef_init). Returns (alpha, coef, result)."""
    basis, n_t, n_x = stepper.basis, grid.n_t, grid.n_x
    K, chol_inv = jnp.asarray(basis.K), jnp.asarray(basis.chol_inv)
    penalty, D_tt = jnp.asarray(library.penalty()), jnp.asarray(second_difference(n_t))
    n_obs, k_obs, y_obs = jnp.asarray(obs.n), jnp.asarray(obs.k), jnp.asarray(obs.y)
    n_w = n_t * n_x

    def loss(p):
        w, coef = p[:n_w].reshape(n_t, n_x), p[n_w:]
        alpha = w @ chol_inv
        u = alpha @ K
        value = weights.data * jnp.sum((u[n_obs, k_obs] - y_obs) ** 2)
        value += weights.dyn * jnp.sum(stepper.residual(alpha, coef) ** 2)
        value += weights.alpha * jnp.sum(w ** 2) + weights.beta * coef @ penalty @ coef
        value += weights.addot / grid.dt ** 4 * jnp.sum((D_tt @ w) ** 2)
        return value

    p0 = np.concatenate([np.asarray(basis.whiten(alpha_init)).ravel(), np.asarray(coef_init)])
    result = lbfgs_jax(loss, p0, maxiter=maxiter)
    return basis.unwhiten(result.x[:n_w].reshape(n_t, n_x)), result.x[n_w:], result


def fit_joint_hard(stepper, library, grid, obs, alpha0_init, coef_init, weights=JointWeights(), max_nfev=100):
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
                                np.sqrt(weights.alpha) * w.ravel(),
                                np.sqrt(weights.beta) * (penalty_chol.T @ p[n_x:]),
                                np.sqrt(weights.addot) / grid.dt ** 2 * (D_tt @ w).ravel()])

    p0 = np.concatenate([np.asarray(basis.whiten(alpha0_init)), np.asarray(coef_init)])
    result = least_squares_jax(residual, p0, max_nfev=max_nfev)
    return np.asarray(trajectory(jnp.asarray(result.x))), result.x[n_x:], result
