"""Step 2 of the two-step method, hard constraint: fit the operator through the rollout, state frozen.

For a fixed atom set (from atom_selection.py), the coefficients b are chosen so that the rollout of the stepper from
alpha_bar_0 matches the frozen Step-1 state at every later column:

    min_b sum_{n >= 1} ||K alpha_bar_n - K Phi^n(alpha_bar_0; b)||^2 + lam_beta sum_m beta_m^T K_m beta_m.

Nonlinear in b through the rollout: trust-region Gauss-Newton with the exact Jacobian (forward mode through the
Newton steps of theta_stepper.py).
"""
import jax.numpy as jnp
import numpy as np

from .jax_optimizers import least_squares_jax


def fit_two_step_operator_hard(stepper, library, alpha_ref, coef_init, lam_beta=1e-6, max_nfev=300):
    """Coefficients of `library` (the active atoms) fitted through the rollout, state frozen at alpha_ref
    (module docstring). `stepper` must use `library.evaluate` as its operator. Returns (coef, scipy result)."""
    basis, n_steps = stepper.basis, alpha_ref.shape[0] - 1
    alpha0, target = jnp.asarray(alpha_ref[0]), basis.values(jnp.asarray(alpha_ref[1:]))
    penalty_chol = jnp.asarray(np.linalg.cholesky(library.penalty() + 1e-10 * np.eye(library.n_coef)))

    def residual(coef):
        rolled = basis.values(stepper.rollout(alpha0, coef, n_steps)[1:])
        return jnp.concatenate([(target - rolled).ravel(), jnp.sqrt(lam_beta) * (penalty_chol.T @ coef)])

    result = least_squares_jax(residual, coef_init, max_nfev=max_nfev)
    return result.x, result
