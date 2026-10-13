"""The four methods as one estimator: state fit -> atom library -> atom selection -> (refit) -> stepper.

    method          stages run by KMDTimeStepper.fit
    two_step_soft   fit_state, fit_two_step_operator_soft                 (Step 1, Step 2 soft + atom selection)
    two_step_hard   ... then refit_two_step_operator_hard                 (coefficients through the rollout)
    one_step_soft   ... two_step_soft, then refit_one_step_soft           (state and coefficients together)
    one_step_hard   ... two_step_soft, then refit_one_step_hard           (alpha_0 and coefficients, state = rollout)

The atoms are selected once, by two_step_soft; the other three methods refit the selected atoms, started from the
two_step_soft fit. Each stage is a plain function in its own module; KMDTimeStepper only wires them together and keeps
the fitted quantities. Fitted attributes end with an underscore.
"""
import jax.numpy as jnp
import numpy as np

from .atom_selection import select_atoms
from .cross_validation import grid_search, heldout_rmse
from .kmd_library import build_library
from .one_step_hard import fit_one_step_hard
from .one_step_soft import OneStepWeights, fit_one_step_soft
from .spacetime_grid import interior_folds
from .state_basis import StateBasis
from .theta_stepper import ThetaStepper
from .two_step_operator_hard import fit_two_step_operator_hard
from .two_step_operator_soft import OperatorRegression
from .two_step_state_fit import fit_state

METHODS = ("two_step_soft", "two_step_hard", "one_step_soft", "one_step_hard")


class KMDTimeStepper:
    """Learn u_t = P(u, u_x, ...) from sparse observations of u on a space-time grid, and step it forward.

    kernel: JaxKernels kernel on scalars for the spatial state basis (max_order times differentiable).
    features, atom_kernels, n_inducing: the operator library (kmd_library.build_library).
    theta, newton_iters: the time stepper (theta_stepper.ThetaStepper).
    lam_u: weight of the RKHS norm of each state column in the state fit.
    """

    def __init__(self, kernel, grid, *, max_order=3, features=None, atom_kernels=None, n_inducing=15, theta=0.5,
                 newton_iters=6, lam_u=1e-6, nugget=1e-8):
        self.grid = grid
        self.basis = StateBasis.build(kernel, grid.x, max_order=max_order, nugget=nugget)
        self.features, self.atom_kernels, self.n_inducing = features, atom_kernels, n_inducing
        self.theta, self.newton_iters, self.lam_u = theta, newton_iters, lam_u

    # ---- one call per method

    def fit(self, obs, method="two_step_soft", *, lam_addot_grid=None, lam_beta_grid=None, n_folds=5, n_blocks=5,
            seed=0, weights=OneStepWeights(), lam_beta_hard=1e-6, **options):
        """Run `method` (one of METHODS, module docstring). `weights` are the one-step weights, `lam_beta_hard` the
        coefficient penalty of two_step_hard; `options` go to the refit's optimiser (maxiter for one_step_soft,
        max_nfev for two_step_hard and one_step_hard)."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.fit_state(obs, lam_addot_grid, n_folds, seed).fit_two_step_operator_soft(lam_beta_grid, n_blocks)
        if method == "two_step_hard":
            self.refit_two_step_operator_hard(lam_beta_hard, **options)
        elif method == "one_step_soft":
            self.refit_one_step_soft(obs, weights, **options)
        elif method == "one_step_hard":
            self.refit_one_step_hard(obs, weights, **options)
        self.method_ = method
        return self

    # ---- the stages

    def fit_state(self, obs, lam_addot_grid=None, n_folds=5, seed=0):
        """Two-step, Step 1: state fit with the time prior, its weight chosen by K-fold CV on the interior
        observations."""
        grid_values = np.r_[0, np.logspace(-12, -1, 23)] if lam_addot_grid is None else np.asarray(lam_addot_grid)
        folds = interior_folds(self.grid, obs, n_folds, seed)
        self.state_cv_ = grid_search(grid_values, lambda lam: heldout_rmse(
            lambda train: fit_state(self.basis, self.grid, train, self.lam_u, lam), self.basis, obs, folds))
        self.alpha_ = fit_state(self.basis, self.grid, obs, self.lam_u, self.state_cv_.best)
        return self

    def fit_two_step_operator_soft(self, lam_beta_grid=None, n_blocks=5):
        """Two-step, Step 2 soft: build the library on the fitted state, select atoms, fit their coefficients."""
        self.full_library_ = build_library(self.basis.jets(jnp.asarray(self.alpha_)), self.features,
                                           self.atom_kernels, self.n_inducing)
        self.regression_ = OperatorRegression.from_state(self.basis, self.full_library_, self.alpha_, self.grid.dt,
                                                         self.theta)
        self.selection_ = select_atoms(self.regression_, lam_beta_grid, n_blocks)
        self._set_operator(self.full_library_.restrict(self.selection_.atoms), self.selection_.coef)
        self.method_ = "two_step_soft"
        return self

    def refit_two_step_operator_hard(self, lam_beta=1e-6, max_nfev=300):
        """Two-step, Step 2 hard: refit the selected coefficients so the rollout from the first column matches the
        fitted state."""
        coef, self.refit_result_ = fit_two_step_operator_hard(self.stepper_, self.library_, self.alpha_, self.coef_,
                                                              lam_beta, max_nfev)
        self.coef_, self.method_ = np.asarray(coef), "two_step_hard"
        return self

    def refit_one_step_soft(self, obs, weights=OneStepWeights(), maxiter=20000):
        """One-step soft on the selected atoms, started from the current fit: all state columns and coefficients."""
        alpha, coef, self.refit_result_ = fit_one_step_soft(self.stepper_, self.library_, self.grid, obs, self.alpha_,
                                                            self.coef_, weights, maxiter)
        self.alpha_, self.coef_, self.method_ = np.asarray(alpha), np.asarray(coef), "one_step_soft"
        return self

    def refit_one_step_hard(self, obs, weights=OneStepWeights(), max_nfev=100):
        """One-step hard on the selected atoms, started from the current fit: alpha_0 and coefficients."""
        alpha, coef, self.refit_result_ = fit_one_step_hard(self.stepper_, self.library_, self.grid, obs,
                                                            self.alpha_[0], self.coef_, weights, max_nfev)
        self.alpha_, self.coef_, self.method_ = np.asarray(alpha), np.asarray(coef), "one_step_hard"
        return self

    def _set_operator(self, library, coef):
        self.library_, self.coef_ = library, np.asarray(coef)
        self.stepper_ = ThetaStepper(self.basis, library.evaluate, self.grid.dt, self.theta, self.newton_iters)

    # ---- using the fit

    def operator(self, jets):
        """P_hat at jets (..., Q + 1) = (u, u_x, ..., d^Q u)."""
        return np.asarray(self.library_.evaluate(jnp.asarray(jets), jnp.asarray(self.coef_)))

    def state(self, x=None, order=0):
        """The fitted state (or its order-th x-derivative) at the time columns: (N_T, len(x)); default x: the nodes."""
        matrix = self.basis.K if x is None and order == 0 else self.basis.matrix(self.grid.x if x is None else x, order)
        return self.alpha_ @ matrix.T

    def rollout(self, u0, n_steps=None, x=None):
        """Step the learned equation forward from nodal values u0 (N_X,) of the initial condition.

        Returns the values at the nodes (or at the points x) for columns 0..n_steps: (n_steps + 1, N_X or len(x)).
        """
        n_steps = self.grid.n_steps if n_steps is None else n_steps
        alpha = np.asarray(self.stepper_.rollout(self.basis.interpolate(u0), jnp.asarray(self.coef_), n_steps))
        return alpha @ (self.basis.K if x is None else self.basis.matrix(x)).T

    def summary(self):
        """The method, the selected atoms and, for the terms linear in a feature, their coefficients in the original
        units."""
        slopes, constant = self.library_.linear_coefficients(self.coef_)
        lines = [f"method: {self.method_}", f"selected atoms: {', '.join(self.library_.names)}",
                 "P_hat(s) = " + " ".join([f"{constant:+.4g}"] + [f"{a:+.4g} {name}" for name, a in slopes.items()])]
        nonlinear = [atom.name for atom in self.library_.atoms if atom.size > 1]
        if nonlinear:
            lines[-1] += " + [" + ", ".join(nonlinear) + "]"
        return "\n".join(lines)
