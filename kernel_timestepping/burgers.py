"""Reference data for the examples and tests: a fine-grid solver for the viscous Burgers-type equation

    u_t = kappa u_xx + alpha u u_x,   x in (0, 1),   u(t, 0) = u(t, 1) = 0,

and a spline of its solution that returns u and its derivatives anywhere. Not part of the method.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import csc_array, diags_array, eye_array
from scipy.sparse.linalg import factorized


def solve_burgers(u0, kappa, alpha, t_final=1.0, dt=1e-4, n_interior=1999, save_every=10):
    """Second-order finite differences in space, Strang splitting in time: half a diffusion step (TR-BDF2), a full
    advection step (explicit midpoint), half a diffusion step.

    u0: callable on an array of x. Returns (t (n_saved,), x (n_interior + 2,), u (n_saved, n_interior + 2)).
    """
    n, h = n_interior, 1.0 / (n_interior + 1)
    x = np.linspace(0, 1, n + 2)
    laplacian = diags_array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], offsets=[-1, 0, 1]) / h ** 2
    gradient = diags_array([-np.ones(n - 1), np.ones(n - 1)], offsets=[-1, 1]) / (2 * h)

    B = laplacian * kappa * (dt / 2)                       # diffusion over half a step
    identity = eye_array(n)
    solve_trapezoid, solve_bdf2 = factorized(csc_array(identity - B / 4)), factorized(csc_array(identity - B / 3))
    forward_trapezoid = identity + B / 4

    def half_diffusion(u):
        u_mid = solve_trapezoid(forward_trapezoid @ u)
        return solve_bdf2((4 * u_mid - u) / 3)

    def advection(u):
        rate = lambda v: alpha * (gradient @ v) * v
        return u + dt * rate(u + dt / 2 * rate(u))

    n_steps = int(round(t_final / dt))
    u = np.asarray(u0(x), dtype=float)[1:-1]
    saved = [u]
    for step in range(1, n_steps + 1):
        u = half_diffusion(advection(half_diffusion(u)))
        if step % save_every == 0:
            saved.append(u)
    t = dt * save_every * np.arange(len(saved))
    return t, x, np.pad(np.array(saved), ((0, 0), (1, 1)))


class ReferenceSolution:
    """Bicubic spline of a solution on a (t, x) grid, with its derivatives.

    Every method takes arrays t (m,) and x (p,) and returns (m, p) (grid=True), or paired points (grid=False).
    """

    def __init__(self, t, x, u):
        self.t, self.x, self.values = np.asarray(t), np.asarray(x), np.asarray(u)
        self._spline = RectBivariateSpline(self.t, self.x, self.values)

    def u(self, t, x, grid=True):
        return self._spline(t, x, grid=grid)

    def u_t(self, t, x, grid=True):
        return self._spline(t, x, dx=1, grid=grid)

    def u_x(self, t, x, grid=True):
        return self._spline(t, x, dy=1, grid=grid)

    def u_xx(self, t, x, grid=True):
        return self._spline(t, x, dy=2, grid=grid)

    def jets(self, t, x, max_order=3):
        """(u, u_x, ..., d^Q u) on the grid t x x: (m, p, Q + 1). Orders above 2 are centred differences of the
        stored solution at the nearest stored time (a cubic spline has no reliable third derivative)."""
        t, x = np.atleast_1d(t), np.atleast_1d(x)
        columns = [self.u(t, x), self.u_x(t, x), self.u_xx(t, x)][:max_order + 1]
        if max_order >= 3:
            rows = self.values[[np.abs(self.t - time).argmin() for time in t]]
            for order in range(1, max_order + 1):
                rows = np.gradient(rows, self.x, axis=1)
                if order >= 3:
                    columns.append(np.stack([np.interp(x, self.x, row) for row in rows]))
        return np.stack(columns, axis=-1)
