"""Lengthscale of the spatial state kernel from the observations.

A Gaussian-process regression of the observations in (t, x) with a diagonally anisotropic Gaussian RBF kernel of unit
variance,

    k((t, x), (t', x')) = exp(-(t - t')^2 / (2 l_t^2) - (x - x')^2 / (2 l_x^2)),

is fitted by maximum marginal likelihood, -log N(y | 0, K + jitter diag(K)), over (l_t, l_x). The spatial state
kernel then uses shrink * l_x (the time lengthscale is not used: the state kernel acts in space only).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jaxkernels import GaussianRBFKernel, hyper, objectives
from scipy.optimize import minimize


def fit_spacetime_lengthscales(tx, y, init=1.2, jitter=1e-7):
    """(l_t, l_x) maximising the marginal likelihood of observations y at points tx (n, 2) = (t, x)."""
    tx, y = jnp.asarray(tx, dtype=float), jnp.asarray(y, dtype=float)
    kernel, names = GaussianRBFKernel(lengthscale=jnp.full(2, float(init))), ["lengthscale"]

    def objective(z):
        return objectives.neg_log_marginal_likelihood(hyper.from_log_vector(kernel, names, z), 0.0, tx, y, jitter)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    result = minimize(lambda z: tuple(np.asarray(v, dtype=float) for v in value_and_grad(jnp.asarray(z))),
                      np.asarray(hyper.to_log_vector(kernel, names)), jac=True, method="L-BFGS-B")
    return np.asarray(hyper.from_log_vector(kernel, names, jnp.asarray(result.x)).lengthscale)


def state_lengthscale(grid, obs, shrink=0.75, **options):
    """shrink * l_x for nodal observations on a grid (see fit_spacetime_lengthscales)."""
    tx = np.stack([grid.t[obs.n], grid.x[obs.k]], axis=1)
    return float(shrink * fit_spacetime_lengthscales(tx, obs.y, **options)[1])
