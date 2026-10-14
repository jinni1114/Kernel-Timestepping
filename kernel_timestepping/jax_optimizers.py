"""Thin wrappers that hand jax objectives (exact derivatives) to scipy optimisers."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize


def stationarity(jac, fun):
    """max_j |J_j . f| / (||J_j|| ||f||): the largest cosine between the residual and a column of the Jacobian.

    Zero at a stationary point of ||f||^2 / 2, and unchanged by rescaling the residual or any parameter, unlike the
    gradient norm (scipy's `optimality`)."""
    norm_f = np.linalg.norm(fun)
    if norm_f == 0:
        return 0.0
    return float(np.max(np.abs(jac.T @ fun) / np.maximum(np.linalg.norm(jac, axis=0) * norm_f, 1e-300)))


def least_squares_jax(residual, x0, max_nfev=300, blowup=1e6, cosine_tol=1e-3):
    """Trust-region Gauss-Newton on a jax residual vector, with its forward-mode Jacobian.

    Non-finite residuals (a rollout that blew up) are replaced by +-blowup, so the trust region rejects the step
    instead of failing.

    scipy reports success whenever a tolerance is met, including `xtol` when the trust region has collapsed far from
    a minimum. The result gets two more fields: `max_cosine` = stationarity(jac, fun) at the end, and
    `converged` = max_cosine <= cosine_tol; a RuntimeWarning is raised when it is False. The default 1e-3 separates
    the fits on the Burgers walkthrough (5 observation draws, two-step and one-step hard): at most 2.8e-4 where
    the coefficients converged, at least 4.0e-3 where the optimiser stopped after 2 to 4 evaluations.
    """
    r, J = jax.jit(residual), jax.jit(jax.jacfwd(residual))

    def fun(x):
        return np.nan_to_num(np.asarray(r(jnp.asarray(x))), nan=blowup, posinf=blowup, neginf=-blowup)

    def jac(x):
        return np.nan_to_num(np.asarray(J(jnp.asarray(x))))

    result = least_squares(fun, np.asarray(x0, dtype=float), jac=jac, method="trf", x_scale="jac", max_nfev=max_nfev)
    result.max_cosine = stationarity(result.jac, result.fun)
    result.converged = result.max_cosine <= cosine_tol
    if not result.converged:
        warnings.warn(f"least squares stopped away from a stationary point ({result.message} after {result.nfev} "
                      f"evaluations; max cosine between residual and Jacobian columns {result.max_cosine:.1e} > "
                      f"{cosine_tol:.0e})", RuntimeWarning, stacklevel=3)
    return result


def lbfgs_jax(loss, x0, maxiter=20000):
    """L-BFGS-B on a jax scalar loss with its reverse-mode gradient. The loss is divided by its starting value, so
    the stopping tolerances do not depend on its scale."""
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    x0 = np.asarray(x0, dtype=float)
    scale = float(loss(jnp.asarray(x0)))
    scale = scale if scale > 0 else 1.0

    def objective(x):
        value, grad = value_and_grad(jnp.asarray(x))
        return float(value) / scale, np.asarray(grad) / scale

    return minimize(objective, x0, jac=True, method="L-BFGS-B",
                    options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-15, gtol=1e-10))
