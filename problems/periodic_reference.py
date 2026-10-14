"""Reference data on a periodic domain: a Fourier pseudo-spectral ETDRK4 solver for equations of the form

    u_t = L u + c_nl (u^2 / 2)_x,     x in [0, P) periodic,

with L a linear differential operator given by its Fourier symbol, and the solution as a Fourier series that returns
u, u_t and any x-derivative anywhere. Not part of the method.

    advection_diffusion(c, nu)       u_t = -c u_x + nu u_xx                   (c_nl = 0)
    kdv(a, b)                        u_t = -a u u_x - b u_xxx
    kuramoto_sivashinsky()           u_t = -u u_x - u_xx - u_xxxx

ETDRK4 as in Kassam & Trefethen (SIAM J. Sci. Comput. 26, 2005), with the phi-functions evaluated by a contour
integral over a full circle (complex symbols); the quadratic term is dealiased by the 2/3 rule.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class PeriodicEquation:
    """u_t = L u + nonlinear * (u^2 / 2)_x; symbol(kappa) is the Fourier symbol of L at wavenumbers kappa."""
    name: str
    symbol: Callable
    nonlinear: float
    terms: dict                 # {feature name: coefficient}, the true operator in the package's feature names


def advection_diffusion(c=1.0, nu=0.05):
    return PeriodicEquation("advection-diffusion", lambda k: -1j * c * k - nu * k ** 2, 0.0, {"u_x": -c, "u_xx": nu})


def kdv(a=6.0, b=1.0):
    return PeriodicEquation("KdV", lambda k: 1j * b * k ** 3, -a, {"u*u_x": -a, "u_xxx": -b})


def kuramoto_sivashinsky():
    return PeriodicEquation("Kuramoto-Sivashinsky", lambda k: k ** 2 - k ** 4, -1.0,
                            {"u*u_x": -1.0, "u_xx": -1.0, "u_xxxx": -1.0})


def wavenumbers(n, period):
    return 2 * np.pi / period * np.fft.rfftfreq(n, 1.0 / n)


def solve_periodic(equation, u0, period, n=256, dt=1e-3, t_final=1.0, save_every=10):
    """ETDRK4 on n Fourier modes. u0: callable on an array of x in [0, period).

    Returns (t (n_saved,), x (n,), u (n_saved, n)), saved every save_every steps."""
    x = period * np.arange(n) / n
    kappa = wavenumbers(n, period)
    L = equation.symbol(kappa).astype(complex)
    dealias = kappa < (2 / 3) * kappa.max()

    def N(v):
        if equation.nonlinear == 0:
            return np.zeros_like(v)
        u = np.fft.irfft(v, n)
        return equation.nonlinear * 0.5j * kappa * np.fft.rfft(u * u) * dealias

    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    # full circle: the half circle + real part of Kassam & Trefethen holds only for real symbols (not KdV, advection)
    roots = np.exp(2j * np.pi * (np.arange(1, 65) - 0.5) / 64)
    LR = dt * L[:, None] + roots[None, :]
    Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, axis=1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1)
    f3 = dt * np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1)

    v = np.fft.rfft(np.asarray(u0(x), dtype=float))
    saved = [np.fft.irfft(v, n)]
    for step in range(1, int(round(t_final / dt)) + 1):
        Nv = N(v)
        a = E2 * v + Q * Nv
        Na = N(a)
        b = E2 * v + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + N(c) * f3
        if step % save_every == 0:
            saved.append(np.fft.irfft(v, n))
    t = dt * save_every * np.arange(len(saved))
    return t, x, np.array(saved)


class PeriodicReferenceSolution:
    """Fourier series in x of a solution saved on a (t, x) grid, with a cubic spline in t of its coefficients.

    Derivatives in x are exact for the series; u_t is the equation's right-hand side evaluated on the series (the
    Nyquist mode is dropped from derivatives). Methods take arrays t (m,) and x (p,) and return (m, p) (grid=True),
    or paired points (grid=False).
    """

    def __init__(self, equation, t, x, u, period):
        self.equation, self.period = equation, float(period)
        self.t, self.x, self.values = np.asarray(t), np.asarray(x), np.asarray(u)
        n = self.values.shape[1]
        self.kappa = wavenumbers(n, self.period)
        weights = np.full(len(self.kappa), 2.0)
        weights[0] = 1.0
        if n % 2 == 0:
            weights[-1] = 0.0                       # Nyquist mode: dropped (it carries no resolved information)
        coef = np.fft.rfft(self.values, axis=1) * weights / n
        self._spline_re, self._spline_im = CubicSpline(self.t, coef.real), CubicSpline(self.t, coef.imag)

    def _coef(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self._spline_re(t) + 1j * self._spline_im(t)

    def _evaluate(self, coef, x, order, grid):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        factor = coef * (1j * self.kappa) ** order
        phase = np.exp(1j * np.outer(x, self.kappa))                        # (p, modes)
        if grid:
            return (factor @ phase.T).real
        return np.einsum("im,im->i", factor, phase).real

    def derivative(self, t, x, order=0, grid=True):
        return self._evaluate(self._coef(t), x, order, grid)

    def u(self, t, x, grid=True):
        return self.derivative(t, x, 0, grid)

    def u_t(self, t, x, grid=True):
        """L u + c_nl u u_x, evaluated on the series."""
        coef = self._coef(t)
        linear = self._evaluate(coef * self.equation.symbol(self.kappa), x, 0, grid)
        if self.equation.nonlinear == 0:
            return linear
        return linear + self.equation.nonlinear * self.u(t, x, grid) * self.derivative(t, x, 1, grid)

    def jets(self, t, x, max_order=4):
        """(u, u_x, ..., d^Q u) on the grid t x x: (m, p, Q + 1)."""
        coef = self._coef(t)
        return np.stack([self._evaluate(coef, x, q, True) for q in range(max_order + 1)], axis=-1)
