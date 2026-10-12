"""Theta-scheme time stepping of u_t = P(u, u_x, ...) in the coefficients of the spatial kernel basis.

Right-hand side on all nodes, zero in the two boundary rows (the boundary values do not move):

    r(alpha; b) = [0; P(s(x_k; alpha); b), k = 1..N_X-2; 0].

Dynamic residual of one step (theta = 1/2: Crank-Nicolson, theta = 1: backward Euler):

    e(alpha_{n+1}, alpha_n; b) = K (alpha_{n+1} - alpha_n) - dt [theta r(alpha_{n+1}; b) + (1 - theta) r(alpha_n; b)].

The step map alpha_{n+1} = Phi(alpha_n; b) solves e = 0 with a fixed number of Newton iterations started at alpha_n,
with Jacobian J = K - dt theta dr/dalpha. A fixed iteration count keeps Phi, and so the rollout, differentiable in
both alpha_0 and b.
"""
import jax
import jax.numpy as jnp


class ThetaStepper:
    """Step map and rollout for the operator P(jets, params) -> values at the interior nodes.

    operator: callable (jets (N_X - 2, Q + 1), params) -> (N_X - 2,), written in jax (e.g. AtomLibrary.evaluate).
    """

    def __init__(self, basis, operator, dt, theta=0.5, newton_iters=6):
        self.basis, self.operator = basis, operator
        self.dt, self.theta, self.newton_iters = float(dt), float(theta), int(newton_iters)
        self._K = jnp.asarray(basis.K)
        self._rollout = jax.jit(self._scan, static_argnums=2)

    def rhs(self, alpha, params):
        """r(alpha; b) for one column alpha (N_X,) or a trajectory (N_T, N_X): P at the interior nodes, zero rows
        at the boundary."""
        values = self.operator(self.basis.jets(alpha), params)
        return jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(1, 1)])

    def residual(self, alphas, params):
        """e for every step of a trajectory: alphas (N_T, N_X) -> (N_T - 1, N_X), row n = e(alpha_{n+1}, alpha_n)."""
        r = self.rhs(alphas, params)
        return (alphas[1:] - alphas[:-1]) @ self._K - self.dt * (self.theta * r[1:] + (1 - self.theta) * r[:-1])

    def step(self, alpha, params):
        """alpha_{n+1} = Phi(alpha_n; b)."""
        explicit = (1 - self.theta) * self.rhs(alpha, params)             # the part of e that Newton does not change
        a = alpha
        for _ in range(self.newton_iters):
            e = self._K @ (a - alpha) - self.dt * (self.theta * self.rhs(a, params) + explicit)
            jacobian = self._K - self.dt * self.theta * jax.jacfwd(self.rhs)(a, params)
            a = a + jnp.linalg.solve(jacobian, -e)
        return a

    def _scan(self, alpha0, params, n_steps):
        _, later = jax.lax.scan(lambda a, _: (self.step(a, params),) * 2, alpha0, None, length=n_steps)
        return jnp.vstack([alpha0[None], later])

    def rollout(self, alpha0, params, n_steps):
        """(n_steps + 1, N_X): row n = Phi^n(alpha_0; b). Differentiable in alpha0 and params."""
        return self._rollout(jnp.asarray(alpha0), params, n_steps)
