"""One-step method, soft constraint: the state and the operator coefficients are fitted together, for a fixed set of
atoms (from atom_selection.py).

Unknowns alpha_0..alpha_{N_T-1} and b,

    J(alpha, b) = lam_d     sum_{(n,k) observed} (y_nk - (K alpha_n)_k)^2
                + lam_dyn   sum_n ||e(alpha_{n+1}, alpha_n; b)||^2                  (all rows of e, theta_stepper.py)
                + lam_alpha sum_n alpha_n^T K alpha_n + lam_beta sum_m beta_m^T K_m beta_m
                + lam_addot / dt^4 sum_n ||alpha_{n+1} - 2 alpha_n + alpha_{n-1}||_K^2,

minimised by L-BFGS-B in the whitened state coordinates w_n = L^T alpha_n. The library (standardisation, inducing
points, active atoms) stays frozen. `OneStepWeights` is shared with the hard constraint (one_step_hard.py), where the
lam_dyn term drops out.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .jax_optimizers import lbfgs_jax
from .two_step_state_fit import second_difference


@dataclass(frozen=True)
class OneStepWeights:
    """Weights of the one-step loss (module docstring)."""
    data: float = 1.0       # lam_d
    dyn: float = 1.0        # lam_dyn (soft constraint only)
    alpha: float = 1e-6     # lam_alpha
    beta: float = 1e-8      # lam_beta
    addot: float = 0.0      # lam_addot


def fit_one_step_soft(stepper, library, grid, obs, alpha_init, coef_init, weights=OneStepWeights(), maxiter=20000):
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
