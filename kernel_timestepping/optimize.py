"""Thin wrappers that hand jax objectives (exact derivatives) to scipy optimisers."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize


def least_squares_jax(residual, x0, max_nfev=300, blowup=1e6):
    """Trust-region Gauss-Newton on a jax residual vector, with its forward-mode Jacobian.

    Non-finite residuals (a rollout that blew up) are replaced by +-blowup, so the trust region rejects the step
    instead of failing."""
    r, J = jax.jit(residual), jax.jit(jax.jacfwd(residual))

    def fun(x):
        return np.nan_to_num(np.asarray(r(jnp.asarray(x))), nan=blowup, posinf=blowup, neginf=-blowup)

    def jac(x):
        return np.nan_to_num(np.asarray(J(jnp.asarray(x))))

    return least_squares(fun, np.asarray(x0, dtype=float), jac=jac, method="trf", x_scale="jac", max_nfev=max_nfev)


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
