"""End-to-end checks of the four methods on the setting of experiments/burgers_walkthrough.ipynb, and of the selection
rule."""
import copy

import jax.numpy as jnp
import numpy as np
import pytest
from jaxkernels import GaussianRBFKernel, LinearKernel, MaternKernel

import kernel_timestepping as kt
from kernel_timestepping.state_lengthscale import state_lengthscale
from problems.burgers_reference import ReferenceSolution, solve_burgers

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


def test_refits_improve_the_two_step_soft_fit(problem, fitted):
    truth, grid, obs = problem
    soft_error = rollout_error(fitted, truth, grid)
    jets_true = truth.jets(grid.t, grid.x_int)

    def uxx_error(alpha):
        return kt.relative_l2(np.asarray(fitted.basis.jets(jnp.asarray(alpha)))[..., 2], jets_true[..., 2])

    two_step_hard = copy.copy(fitted).refit_two_step_operator_hard()
    assert two_step_hard.refit_result_.success and two_step_hard.method_ == "two_step_hard"
    assert rollout_error(two_step_hard, truth, grid) < 0.5 * soft_error

    one_step_soft = copy.copy(fitted).refit_one_step_soft(obs, kt.OneStepWeights(dyn=1.0), lam_dyn_grid=None,
                                                          lam_addot_grid_one_step=None, maxiter=4000)
    assert uxx_error(one_step_soft.alpha_) < 0.5 * uxx_error(fitted.alpha_)
    assert rollout_error(one_step_soft, truth, grid) < 0.5 * soft_error


def test_one_step_hard_state_is_the_rollout_of_its_first_column(problem, fitted):
    _, _, obs = problem
    one_step_hard = copy.copy(fitted).refit_one_step_hard(obs, max_nfev=5)
    rolled = one_step_hard.stepper_.rollout(jnp.asarray(one_step_hard.alpha_[0]), jnp.asarray(one_step_hard.coef_),
                                            one_step_hard.grid.n_steps)
    assert np.allclose(one_step_hard.alpha_, np.asarray(rolled))
    assert one_step_hard.library_.names == fitted.library_.names          # the atom set stays fixed


def test_one_step_hard_with_default_weights_recovers_the_equation(problem, fitted):
    """The state norm is on alpha_0 only: a norm on the rolled-out columns biased u*u_x to about 0.49 here."""
    truth, grid, obs = problem
    one_step_hard = copy.copy(fitted).refit_one_step_hard(obs)
    slopes, _ = one_step_hard.library_.linear_coefficients(one_step_hard.coef_)
    assert abs(slopes["u_xx"] - KAPPA) < 0.01 * KAPPA and abs(slopes["u*u_x"] - ALPHA) < 0.01 * ALPHA
    assert rollout_error(one_step_hard, truth, grid) < 0.1 * rollout_error(fitted, truth, grid)


def test_one_step_soft_cross_validates_lam_dyn_then_lam_addot(problem, fitted):
    _, _, obs = problem
    model = copy.copy(fitted).refit_one_step_soft(obs, lam_dyn_grid=[0.0, 1.0], lam_addot_grid_one_step=[0.0],
                                                  cv_maxiter=300, maxiter=300)
    dyn, addot = model.one_step_cv_["dyn"], model.one_step_cv_["addot"]
    assert dyn.fold_scores.shape == (2, 5) and dyn.best == 1.0                   # the dynamics help on held-out data
    assert set(addot.values) == {0.0, fitted.state_cv_.best}                     # the Step-1 value is always a candidate
    assert model.weights_.dyn == dyn.best and model.weights_.addot == addot.best


def test_fit_method_runs_the_same_stages(problem, fitted):
    _, grid, obs = problem
    model = kt.KMDTimeStepper(fitted.basis.kernel, grid)
    model.fit(obs, method="two_step_hard", lam_addot_grid=[0.0, 1e-10, 1e-8, 1e-6])
    staged = copy.copy(fitted).refit_two_step_operator_hard()
    assert model.method_ == "two_step_hard" and np.allclose(model.coef_, staged.coef_)
    with pytest.raises(ValueError):
        model.fit(obs, method="three_step")


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
