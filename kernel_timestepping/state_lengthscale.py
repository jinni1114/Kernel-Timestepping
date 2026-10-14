"""Lengthscale of the spatial state kernel from the observations.

A Gaussian-process regression of the observations in (t, x) with a diagonally anisotropic Gaussian RBF kernel of unit
variance,

    k((t, x), (t', x')) = exp(-(t - t')^2 / (2 l_t^2) - (x - x')^2 / (2 l_x^2)),

is fitted by maximum marginal likelihood, -log N(y | 0, K + jitter diag(K)), over (l_t, l_x). The spatial state
kernel then uses shrink * l_x (the time lengthscale is not used: the state kernel acts in space only).

A smooth kernel at that lengthscale can make K numerically singular (the periodic Gaussian kernel on 32 nodes over
[0, 2 pi) at the fitted lengthscale gives cond(K) ~ 2e18, and every rollout blows up). `max_cond` caps cond(K): the
lengthscale is then the largest one, at most shrink * l_x, with cond(K) <= max_cond (bisection in log l).

On a periodic grid the x-factor is periodic (jaxkernels PeriodicKernel: (x - x')^2 is replaced by the squared chordal
distance (P / pi)^2 sin^2(pi (x - x') / P), the same for |x - x'| << P), so l_x is in the same units.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jaxkernels import GaussianRBFKernel, PeriodicKernel, hyper, objectives
from scipy.optimize import minimize


def fit_spacetime_lengthscales(tx, y, init=1.2, jitter=1e-7, period=None):
    """(l_t, l_x) maximising the marginal likelihood of observations y at points tx (n, 2) = (t, x); period: the
    period in x, None for a non-periodic x."""
    tx, y = jnp.asarray(tx, dtype=float), jnp.asarray(y, dtype=float)
    if period is None:
        kernel = GaussianRBFKernel(lengthscale=jnp.full(2, float(init)))
    else:       # periodic in x; t gets a period so long that its factor is the Gaussian one
        t_period = 1e3 * float(jnp.ptp(tx[:, 0]) + 1.0)
        kernel = PeriodicKernel(period=jnp.array([t_period, float(period)]), lengthscale=jnp.full(2, float(init)))
    names = ["lengthscale"]                # the periods stay fixed

    def objective(z):
        return objectives.neg_log_marginal_likelihood(hyper.from_log_vector(kernel, names, z), 0.0, tx, y, jitter)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    result = minimize(lambda z: tuple(np.asarray(v, dtype=float) for v in value_and_grad(jnp.asarray(z))),
                      np.asarray(hyper.to_log_vector(kernel, names)), jac=True, method="L-BFGS-B")
    return np.asarray(hyper.from_log_vector(kernel, names, jnp.asarray(result.x)).lengthscale)


def state_lengthscale(grid, obs, shrink=0.75, max_cond=None, kernel=None, **options):
    """shrink * l_x for nodal observations on a grid (see fit_spacetime_lengthscales).

    max_cond: if given, the largest lengthscale <= shrink * l_x with cond(K) <= max_cond for the state kernel
    kernel(lengthscale) (default: GaussianRBFKernel, or PeriodicKernel with the grid's period)."""
    tx = np.stack([grid.t[obs.n], grid.x[obs.k]], axis=1)
    ell = float(shrink * fit_spacetime_lengthscales(tx, obs.y, period=grid.period, **options)[1])
    if max_cond is None:
        return ell
    if kernel is None:
        kernel = ((lambda l: GaussianRBFKernel(lengthscale=l)) if grid.period is None
                  else (lambda l: PeriodicKernel(period=grid.period, lengthscale=l)))
    return capped_lengthscale(kernel, grid.x, ell, max_cond)


def capped_lengthscale(kernel, x, ell, max_cond, iterations=40):
    """The largest lengthscale l <= ell with cond(K) <= max_cond, K = [kernel(l)(x_k, x_l)]; bisection in log l."""
    def cond(l):
        k = kernel(l)
        K = np.asarray(jax.vmap(lambda a: jax.vmap(lambda b: k(a, b))(jnp.asarray(x)))(jnp.asarray(x)))
        return np.linalg.cond(K)

    if cond(ell) <= max_cond:
        return float(ell)
    lo, hi = np.log(ell) - np.log(1e3), np.log(ell)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cond(np.exp(mid)) <= max_cond else (lo, mid)
    return float(np.exp(lo))
