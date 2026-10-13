"""Unit checks of the building blocks: state basis, atom library, state fit, time stepper."""
import jax.numpy as jnp
import numpy as np
import pytest
from jaxkernels import GaussianRBFKernel

import kernel_timestepping as kt

KAPPA, ALPHA = 0.01, 1.0


@pytest.fixture(scope="module")
def grid():
    return kt.SpaceTimeGrid(kt.blended_chebyshev_nodes(21), dt=0.05, n_t=9)


@pytest.fixture(scope="module")
def basis(grid):
    return kt.StateBasis.build(GaussianRBFKernel(lengthscale=0.12), grid.x)


def smooth_field(t, x):
    """A smooth field that vanishes on the boundary; t (m,), x (p,) -> (m, p)."""
    return np.exp(-t)[:, None] * np.sin(np.pi * x)[None, :] + 0.3 * np.cos(t)[:, None] * np.sin(2 * np.pi * x)[None, :]


def test_nodes_include_end_points():
    x = kt.blended_chebyshev_nodes(31)
    assert x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) > 0)


def test_derivative_matrices_match_finite_differences(basis, grid):
    h = 1e-5
    for order in range(1, 4):
        lower = basis.matrix(grid.x_int - h, order - 1)
        upper = basis.matrix(grid.x_int + h, order - 1)
        finite_difference = (upper - lower) / (2 * h)
        assert np.abs(basis.D[order] - finite_difference).max() / np.abs(basis.D[order]).max() < 1e-6
    assert np.allclose(basis.D[0], basis.K[1:-1])


def test_interpolation_and_whitening(basis, grid):
    u = smooth_field(grid.t, grid.x)
    alpha = basis.interpolate(u)
    assert np.allclose(basis.values(alpha), u, atol=1e-8)
    assert np.allclose(basis.unwhiten(basis.whiten(alpha)), alpha)
    w = basis.whiten(alpha)                                   # ||w||^2 = alpha^T (K + nugget I) alpha
    regularised_norm = np.einsum("nk,kl,nl->n", alpha, basis.K + 1e-8 * np.eye(grid.n_x), alpha)
    assert np.allclose(np.sum(w ** 2, axis=1), regularised_norm)


def test_jets_are_derivatives_of_the_state(basis, grid):
    u = smooth_field(grid.t, grid.x)
    jets = np.asarray(basis.jets(basis.interpolate(u)))
    exact_ux = (np.exp(-grid.t)[:, None] * np.pi * np.cos(np.pi * grid.x_int)
                + 0.3 * np.cos(grid.t)[:, None] * 2 * np.pi * np.cos(2 * np.pi * grid.x_int))
    assert jets.shape == (grid.n_t, grid.n_x - 2, 4)
    assert kt.relative_l2(jets[..., 1], exact_ux) < 1e-2


def test_library_atoms(basis, grid):
    jets = basis.jets(basis.interpolate(smooth_field(grid.t, grid.x)))
    library = kt.build_library(jets, n_inducing=7)
    assert len(library.atoms) == 1 + 3 * 6 and library.n_coef == 1 + 6 + 2 * 6 * 7
    design = np.asarray(library.design(jets))
    z = np.asarray(library.standardize(jets))
    offsets = library.offsets
    assert np.allclose(design[..., 0], 1.0)                                         # constant atom
    linear = library.names.index("linear[u_xx]")
    assert np.allclose(design[..., offsets[linear]], z[..., 2])                     # linear atom: phi(z) = z
    penalty = library.penalty()
    assert np.allclose(penalty, penalty.T) and np.linalg.eigvalsh(penalty).min() > -1e-10
    assert np.isclose(penalty[offsets[linear], offsets[linear]], 1.0)

    subset = [0, linear, library.names.index("rbf[u]")]
    restricted = library.restrict(subset)
    assert np.allclose(np.asarray(restricted.design(jets)), design[..., library.columns(subset)])


def test_library_represents_the_burgers_operator(basis, grid):
    jets = np.asarray(basis.jets(basis.interpolate(smooth_field(grid.t, grid.x))))
    library = kt.build_library(jets)
    names = library.names
    active = library.restrict([names.index("const"), names.index("linear[u_xx]"), names.index("linear[u*u_x]")])
    coef = np.array([KAPPA * library.center[2] + ALPHA * library.center[4],
                     KAPPA * library.scale[2], ALPHA * library.scale[4]])
    exact = KAPPA * jets[..., 2] + ALPHA * jets[..., 0] * jets[..., 1]
    assert np.allclose(np.asarray(active.evaluate(jnp.asarray(jets), jnp.asarray(coef))), exact)
    slopes, constant = active.linear_coefficients(coef)
    assert np.isclose(slopes["u_xx"], KAPPA) and np.isclose(slopes["u*u_x"], ALPHA) and abs(constant) < 1e-12


def test_ridge_svd_solves_the_normal_equations():
    rng = np.random.default_rng(0)
    M, y, reg = rng.standard_normal((30, 12)), rng.standard_normal(30), 1e-2
    assert np.allclose(kt.ridge_svd(M, y, reg), np.linalg.solve(M.T @ M + reg * np.eye(12), M.T @ y))


def test_state_fit_without_time_prior_decouples_columns(basis, grid):
    rng = np.random.default_rng(1)
    n, k = np.meshgrid(np.arange(grid.n_t), np.arange(grid.n_x), indexing="ij")
    keep = rng.random(n.size) < 0.6
    obs = kt.NodalObservations(n.ravel()[keep], k.ravel()[keep], smooth_field(grid.t, grid.x).ravel()[keep])
    alpha = kt.fit_state(basis, grid, obs, lam_u=1e-6, lam_addot=0.0)
    column = 3
    rows = obs.n == column
    design = basis.K[obs.k[rows]] @ basis.chol_inv.T
    w = np.linalg.solve(design.T @ design + 1e-6 * np.eye(grid.n_x), design.T @ obs.y[rows])
    assert np.allclose(alpha[column], basis.unwhiten(w), atol=1e-7)


def test_state_fit_recovers_a_densely_observed_field(basis, grid):
    u = smooth_field(grid.t, grid.x)
    n, k = np.meshgrid(np.arange(grid.n_t), np.arange(grid.n_x), indexing="ij")
    obs = kt.NodalObservations(n.ravel(), k.ravel(), u.ravel())
    for lam_addot in (0.0, 1e-10):
        alpha = kt.fit_state(basis, grid, obs, lam_addot=lam_addot)
        assert kt.relative_l2(basis.values(alpha), u) < 1e-3


def test_theta_step_of_a_linear_operator_is_one_linear_solve(basis, grid):
    """For P = kappa u_xx the residual is linear in alpha_{n+1}, so Newton converges in one iteration to
    (K - dt theta R) alpha_{n+1} = (K + dt (1 - theta) R) alpha_n, R = kappa [0; D_2; 0]."""
    stepper = kt.ThetaStepper(basis, lambda jets, kappa: kappa * jets[..., 2], grid.dt, theta=0.5)
    alpha = basis.interpolate(smooth_field(grid.t[:1], grid.x)[0])
    R = KAPPA * np.pad(basis.D[2], ((1, 1), (0, 0)))
    expected = np.linalg.solve(basis.K - grid.dt * 0.5 * R, (basis.K + grid.dt * 0.5 * R) @ alpha)
    assert np.allclose(np.asarray(stepper.step(jnp.asarray(alpha), KAPPA)), expected, rtol=1e-6, atol=1e-8)


def test_rollout_satisfies_the_dynamic_residual_and_keeps_the_boundary(basis, grid):
    burgers = lambda jets, params: params[0] * jets[..., 2] + params[1] * jets[..., 0] * jets[..., 1]
    stepper = kt.ThetaStepper(basis, burgers, grid.dt)
    params = jnp.array([KAPPA, ALPHA])
    alpha0 = basis.interpolate(smooth_field(grid.t[:1], grid.x)[0])
    trajectory = stepper.rollout(alpha0, params, grid.n_steps)
    assert trajectory.shape == (grid.n_t, grid.n_x)
    assert np.abs(np.asarray(stepper.residual(trajectory, params))).max() < 1e-9
    u = np.asarray(basis.values(trajectory))
    assert np.abs(u[:, [0, -1]] - u[0, [0, -1]]).max() < 1e-9
