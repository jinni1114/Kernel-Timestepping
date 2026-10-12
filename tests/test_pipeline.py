"""End-to-end checks on the setting of examples/kmd_fit_and_rollout.ipynb, and of the selection rule."""
import copy

import jax.numpy as jnp
import numpy as np
import pytest
from jaxkernels import GaussianRBFKernel, LinearKernel, MaternKernel

import kernel_timestepping as kt
from kernel_timestepping.burgers import ReferenceSolution, solve_burgers
from kernel_timestepping.lengthscale import state_lengthscale

KAPPA, ALPHA = 0.01, 1.0


@pytest.fixture(scope="module")
def problem():
    modes = np.arange(1, 6)
    u0 = lambda x: np.sin(np.pi * np.outer(x, modes)) @ np.array([0.06, 0.08, 0.12, 0.075, 0.025])
    truth = ReferenceSolution(*solve_burgers(u0, KAPPA, ALPHA, t_final=1.0))
    grid = kt.SpaceTimeGrid(kt.blended_chebyshev_nodes(31), dt=0.04, n_t=26)
    obs = kt.sample_nodal_observations(grid, lambda t, x: truth.u(t, x, grid=False), n_interior=100, seed=0)
    return truth, grid, obs


@pytest.fixture(scope="module")
def fitted(problem):
    _, grid, obs = problem
    model = kt.KMDTimeStepper(GaussianRBFKernel(lengthscale=state_lengthscale(grid, obs)), grid)
    return model.fit(obs, lam_addot_grid=[0.0, 1e-10, 1e-8, 1e-6])


def rollout_error(model, truth, grid):
    u_true = truth.u(grid.t, grid.x)
    return kt.relative_l2(model.rollout(u_true[0]), u_true)


def test_observations_cover_initial_column_and_boundary(problem):
    _, grid, obs = problem
    assert len(obs) == 181 and obs.is_interior(grid).sum() == 100 and obs.is_boundary(grid).sum() == 52
    assert len(set(zip(obs.n, obs.k))) == len(obs)
    folds = kt.interior_folds(grid, obs, 5)
    assert sorted(np.concatenate(folds)) == sorted(np.flatnonzero(obs.is_interior(grid)))


def test_two_step_fit_recovers_the_equation(problem, fitted):
    truth, grid, _ = problem
    assert fitted.library_.names == ["const", "linear[u_xx]", "linear[u*u_x]"]
    slopes, constant = fitted.library_.linear_coefficients(fitted.coef_)
    assert abs(slopes["u_xx"] - KAPPA) < 0.1 * KAPPA and abs(slopes["u*u_x"] - ALPHA) < 0.1 * ALPHA
    assert abs(constant) < 5e-3
    assert rollout_error(fitted, truth, grid) < 0.03
    assert fitted.state_cv_.best_score < 0.1 * fitted.state_cv_.scores[0]      # the time prior helps


def test_operator_matches_library_evaluation(problem, fitted):
    truth, grid, _ = problem
    jets = truth.jets(grid.t, grid.x_int)
    slopes, constant = fitted.library_.linear_coefficients(fitted.coef_)
    by_hand = constant + slopes["u_xx"] * jets[..., 2] + slopes["u*u_x"] * jets[..., 0] * jets[..., 1]
    assert np.allclose(fitted.operator(jets), by_hand)


def test_refinements_improve_the_two_step_fit(problem, fitted):
    truth, grid, obs = problem
    soft_error = rollout_error(fitted, truth, grid)
    jets_true = truth.jets(grid.t, grid.x_int)

    def uxx_error(alpha):
        return kt.relative_l2(np.asarray(fitted.basis.jets(jnp.asarray(alpha)))[..., 2], jets_true[..., 2])

    through_rollout = copy.copy(fitted).refine_through_rollout()
    assert through_rollout.rollout_fit_.success
    assert rollout_error(through_rollout, truth, grid) < 0.5 * soft_error

    joint = copy.copy(fitted).refine_jointly(obs, kt.JointWeights(dyn=1.0), maxiter=4000)
    assert uxx_error(joint.alpha_) < 0.5 * uxx_error(fitted.alpha_)
    assert rollout_error(joint, truth, grid) < 0.5 * soft_error


def test_one_standard_error_rule_picks_the_smallest_set_within_one_se():
    def step(n_atoms, cv, se):
        return kt.AtomSet(tuple(range(n_atoms + 1)), cv, se, 1e-3, np.array([cv]), np.array([[cv]]))

    path = [step(5, 1.30, 0.1), step(4, 1.00, 0.1), step(3, 1.05, 0.1), step(2, 1.09, 0.1), step(1, 2.0, 0.3)]
    i_min, i_sel, threshold = kt.one_standard_error(path)
    assert (i_min, i_sel) == (1, 3) and np.isclose(threshold, 1.10)


def test_other_kernels_plug_in(problem):
    """A Matern state kernel and a custom atom-kernel dictionary go through the same code path."""
    truth, grid, obs = problem
    basis = kt.StateBasis.build(MaternKernel(p=4, lengthscale=0.2), grid.x)
    assert np.isfinite(basis.D).all()
    alpha = kt.fit_state(basis, grid, obs, lam_addot=1e-8)
    assert kt.relative_l2(basis.values(alpha), truth.u(grid.t, grid.x)) < 0.05

    library = kt.build_library(basis.jets(alpha), kernels={"linear": LinearKernel(), "matern52": MaternKernel(p=2)},
                               n_inducing=8)
    assert library.n_coef == 1 + 6 + 6 * 8
    regression = kt.OperatorRegression.from_state(basis, library, alpha, grid.dt)
    coef = regression.fit(range(len(library.atoms)), lam_beta=1e-3)
    assert np.isfinite(coef).all() and np.isfinite(regression.residual(range(len(library.atoms)), coef)).all()
