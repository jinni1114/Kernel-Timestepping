"""Kernel time-stepping equation learning with a KMD operator library.

Learns an evolution equation u_t = P(u, u_x, u_xx, ...) from sparse observations of u: the state is a spatial kernel
expansion at every time column, the operator is a sum of kernel modes on features of the state, and the two are tied
by a theta-scheme time step. See docs/algorithm.md for the algorithm and experiments/ for a walk-through.

Modules, in pipeline order:
    spacetime_grid, state_basis, state_lengthscale, kmd_library, theta_stepper     building blocks
    two_step_state_fit, two_step_operator_soft, atom_selection, two_step_operator_hard   two-step method
    one_step_soft, one_step_hard                                                    one-step methods
    cross_validation, jax_optimizers                                                utilities
    pipeline                                                                        KMDTimeStepper.fit(method=...)
"""
import jaxkernels as _jaxkernels  # noqa: F401  (enables float64 in JAX, which the kernel matrices need)

from .atom_selection import AtomSelection, AtomSet, backward_elimination, one_standard_error, score_atom_set, select_atoms
from .cross_validation import GridSearch, grid_search, heldout_rmse, relative_l2, rms
from .kmd_library import Atom, AtomLibrary, Feature, build_library, default_atom_kernels, default_features
from .one_step_hard import fit_one_step_hard
from .one_step_soft import OneStepWeights, fit_one_step_soft
from .pipeline import METHODS, KMDTimeStepper
from .spacetime_grid import (NodalObservations, SpaceTimeGrid, blended_chebyshev_nodes, interior_folds,
                             sample_nodal_observations, time_blocks)
from .state_basis import StateBasis, collocation_matrix
from .state_lengthscale import fit_spacetime_lengthscales, state_lengthscale
from .theta_stepper import ThetaStepper
from .two_step_operator_hard import fit_two_step_operator_hard
from .two_step_operator_soft import OperatorRegression
from .two_step_state_fit import fit_state, ridge_svd, second_difference

__all__ = [
    "Atom", "AtomLibrary", "AtomSelection", "AtomSet", "Feature", "GridSearch", "KMDTimeStepper", "METHODS",
    "NodalObservations", "OneStepWeights", "OperatorRegression", "SpaceTimeGrid", "StateBasis", "ThetaStepper",
    "backward_elimination", "blended_chebyshev_nodes", "build_library", "collocation_matrix",
    "default_atom_kernels", "default_features", "fit_one_step_hard", "fit_one_step_soft",
    "fit_spacetime_lengthscales", "fit_state", "fit_two_step_operator_hard", "grid_search", "heldout_rmse",
    "interior_folds", "one_standard_error", "relative_l2", "ridge_svd", "rms", "sample_nodal_observations",
    "score_atom_set", "second_difference", "select_atoms", "state_lengthscale", "time_blocks",
]
