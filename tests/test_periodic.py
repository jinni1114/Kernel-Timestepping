"""Periodic domains: grid and basis, the condition-capped lengthscale, the spectral reference solver, and an
end-to-end fit on advection-diffusion."""
import numpy as np
import pytest
from jaxkernels import PeriodicKernel

import kernel_timestepping as kt
from problems.periodic_reference import (PeriodicReferenceSolution, advection_diffusion, kdv, solve_periodic,
                                         wavenumbers)

PERIOD = 2 * np.pi


@pytest.fixture(scope="module")
def grid():
    return kt.SpaceTimeGrid(kt.periodic_nodes(32, PERIOD), dt=0.04, n_t=26, period=PERIOD)


def test_periodic_grid_has_no_boundary(grid):
    assert grid.periodic and np.array_equal(grid.collocation, np.arange(32)) and grid.boundary.size == 0
    obs = kt.sample_nodal_observations(grid, lambda t, x: np.sin(x - t), n_interior=50, seed=0)
    assert len(obs) == 32 + 50 and not obs.is_boundary(grid).any()
    with pytest.raises(ValueError):
        kt.SpaceTimeGrid(np.linspace(0, PERIOD, 32), dt=0.04, n_t=26, period=PERIOD)    # end point included


def test_periodic_basis_collocates_derivatives_at_every_node(grid):
    basis = kt.StateBasis.build(PeriodicKernel(period=PERIOD, lengthscale=0.6), grid, max_order=4)
    assert basis.D.shape == (5, 32, 32)
    jets = np.asarray(basis.jets(basis.interpolate(np.sin(2 * grid.x))))
    exact = np.stack([np.sin(2 * grid.x), 2 * np.cos(2 * grid.x), -4 * np.sin(2 * grid.x), -8 * np.cos(2 * grid.x),
                      16 * np.sin(2 * grid.x)], axis=-1)
    assert np.abs(jets - exact).max() < 1e-4 * 16


def test_capped_lengthscale_meets_the_condition_bound(grid):
    kernel = lambda ell: PeriodicKernel(period=PERIOD, lengthscale=ell)
    ell = kt.capped_lengthscale(kernel, grid.x, 1.5, max_cond=1e10)
    K = kt.StateBasis.build(kernel(ell), grid).K
    assert ell < 1.5 and 1e9 < np.linalg.cond(K) <= 1e10
    assert kt.capped_lengthscale(kernel, grid.x, 0.3, max_cond=1e10) == 0.3      # already below the bound


def test_spectral_reference_is_exact_on_known_solutions():
    equation = advection_diffusion(1.0, 0.05)
    u0 = lambda x: np.exp(np.sin(x)) - 1
    t, x, u = solve_periodic(equation, u0, PERIOD, n=64, dt=1e-3, t_final=0.5, save_every=50)
    exact = np.fft.irfft(np.fft.rfft(u0(x)) * np.exp(equation.symbol(wavenumbers(64, PERIOD)) * 0.5), 64)
    assert np.abs(u[-1] - exact).max() < 1e-12

    period, c = 30.0, 1.0                                               # one KdV soliton, speed c
    soliton = lambda x, s: c / 2 / np.cosh(np.sqrt(c) / 2 * (((x - c * s - 10) + period / 2) % period - period / 2)) ** 2
    t, x, u = solve_periodic(kdv(), lambda x: soliton(x, 0), period, n=128, dt=1e-3, t_final=2.0, save_every=100)
    assert np.abs(u[-1] - soliton(x, 2.0)).max() < 1e-6
    reference = PeriodicReferenceSolution(kdv(), t, x, u, period)
    xs, ts, h = np.linspace(0, period, 7)[:-1], np.array([0.55, 1.3]), 1e-5
    assert np.allclose(reference.u_t(ts, xs), (reference.u(ts + h, xs) - reference.u(ts - h, xs)) / (2 * h), atol=1e-4)


def test_two_step_fit_on_advection_diffusion(grid):
    equation = advection_diffusion(1.0, 0.05)
    t, x, u = solve_periodic(equation, lambda x: np.exp(np.sin(x)) - 1 + 0.3 * np.cos(2 * x), PERIOD, n=128,
                             dt=1e-3, t_final=1.1, save_every=10)
    truth = PeriodicReferenceSolution(equation, t, x, u, PERIOD)
    obs = kt.sample_nodal_observations(grid, lambda t, x: truth.u(t, x, grid=False), n_interior=150, seed=0)
    ell = kt.state_lengthscale(grid, obs, max_cond=1e12)
    model = kt.KMDTimeStepper(PeriodicKernel(period=PERIOD, lengthscale=ell), grid).fit(obs)
    slopes, _ = model.library_.linear_coefficients(model.coef_)
    assert abs(slopes["u_x"] + 1.0) < 1e-3
    exact = truth.u(grid.t, grid.x)
    assert kt.relative_l2(model.rollout(exact[0]), exact) < 2e-3
