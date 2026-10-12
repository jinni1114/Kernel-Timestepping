"""Kernel time-stepping equation learning with a KMD operator library.

Learns an evolution equation u_t = P(u, u_x, u_xx, ...) from sparse observations of u: the state is a spatial kernel
expansion at every time column, the operator is a sum of kernel modes on features of the state, and the two are tied
by a theta-scheme time step. See KMD_ALGORITHM.md for the algorithm and examples/ for a walk-through.
"""
import jaxkernels as _jaxkernels  # noqa: F401  (enables float64 in JAX, which the kernel matrices need)

from .cv import GridSearch, grid_search, heldout_rmse, relative_l2, rms
from .dynamics import ThetaStepper
from .grid import (NodalObservations, SpaceTimeGrid, blended_chebyshev_nodes, interior_folds,
                   sample_nodal_observations, time_blocks)
from .joint_fit import JointWeights, fit_joint_hard, fit_joint_soft
from .lengthscale import fit_spacetime_lengthscales, state_lengthscale
from .library import Atom, AtomLibrary, Feature, build_library, default_atom_kernels, default_features
from .model import KMDTimeStepper
from .operator_fit import (AtomSelection, AtomSet, OperatorRegression, backward_elimination,
                           fit_operator_through_rollout, one_standard_error, score_atom_set, select_atoms)
from .state import StateBasis, collocation_matrix
from .state_fit import fit_state, ridge_svd, second_difference

__all__ = [
    "Atom", "AtomLibrary", "AtomSelection", "AtomSet", "Feature", "GridSearch", "JointWeights", "KMDTimeStepper",
    "NodalObservations", "OperatorRegression", "SpaceTimeGrid", "StateBasis", "ThetaStepper",
    "backward_elimination", "blended_chebyshev_nodes", "build_library", "collocation_matrix",
    "default_atom_kernels", "default_features", "fit_joint_hard", "fit_joint_soft",
    "fit_operator_through_rollout", "fit_spacetime_lengthscales", "fit_state", "grid_search", "heldout_rmse",
    "interior_folds", "one_standard_error", "relative_l2", "ridge_svd", "rms", "sample_nodal_observations",
    "score_atom_set", "second_difference", "select_atoms", "state_lengthscale", "time_blocks",
]
