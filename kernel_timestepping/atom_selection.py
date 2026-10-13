"""Atom selection for the operator, on the Step-2 soft regression (two_step_operator_soft.py).

lam_beta is chosen by cross-validation over contiguous blocks of time steps (score: RMS of e on the held-out steps).
Atoms are selected by backward elimination: starting from all atoms, drop at each step the atom whose removal gives
the lowest CV score (lam_beta re-chosen for every candidate set), down to one prunable atom. Among the nested sets on
that path, the one-standard-error rule picks the smallest set whose CV score is within one standard error of the best
score on the path.

The selection is made once; the two-step hard and both one-step methods refit the coefficients of the selected atoms.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .spacetime_grid import time_blocks


@dataclass(frozen=True)
class AtomSet:
    """An atom set scored by time-block CV, with lam_beta chosen on the grid."""
    atoms: tuple                # atom indices in the full library
    cv: float                   # pooled CV score at the chosen lam_beta
    se: float                   # standard error of the block scores at the chosen lam_beta
    lam_beta: float
    cv_curve: np.ndarray        # (n_lam,) pooled CV score along the lam_beta grid
    cv_blocks: np.ndarray       # (n_lam, n_blocks)
    dropped: Optional[int] = None    # the atom removed from this set at the next step of the path


def score_atom_set(regression, atoms, lam_grid, blocks):
    cv_blocks, cv_curve = regression.cv(atoms, lam_grid, blocks)
    best = int(np.argmin(cv_curve))
    return AtomSet(tuple(atoms), float(cv_curve[best]), float(cv_blocks[best].std(ddof=1) / np.sqrt(len(blocks))),
                   float(lam_grid[best]), cv_curve, cv_blocks)


def backward_elimination(regression, lam_grid, blocks):
    """Path of nested atom sets, from all atoms down to one prunable atom (non-prunable atoms always stay)."""
    library = regression.library
    current = score_atom_set(regression, range(len(library.atoms)), lam_grid, blocks)
    path = []
    while True:
        prunable = [m for m in current.atoms if library.atoms[m].prunable]
        if len(prunable) <= 1:
            path.append(current)
            return path
        candidates = [(score_atom_set(regression, [a for a in current.atoms if a != m], lam_grid, blocks), m)
                      for m in prunable]
        best, dropped = min(candidates, key=lambda c: (c[0].cv, c[1]))
        path.append(replace(current, dropped=dropped))
        current = best


def one_standard_error(path):
    """(i_min, i_sel, threshold): i_min has the lowest CV score on the path; i_sel is the last (smallest) set with
    cv <= threshold = cv_min + se_min."""
    i_min = int(np.argmin([step.cv for step in path]))
    threshold = path[i_min].cv + path[i_min].se
    i_sel = max(i for i, step in enumerate(path) if step.cv <= threshold)
    return i_min, i_sel, threshold


@dataclass(frozen=True)
class AtomSelection:
    """Result of select_atoms: the elimination path, the selected set and its fitted coefficients."""
    path: list
    i_min: int
    i_sel: int
    threshold: float
    lam_grid: np.ndarray
    coef: np.ndarray            # coefficients of the selected atoms, stacked in the order of `atoms`

    @property
    def selected(self):
        return self.path[self.i_sel]

    @property
    def atoms(self):
        return self.selected.atoms

    @property
    def lam_beta(self):
        return self.selected.lam_beta


def select_atoms(regression, lam_grid=None, n_blocks=5):
    """Backward elimination + one-standard-error rule, then the fit of the selected set on all time steps."""
    lam_grid = np.logspace(-12, 2, 15) if lam_grid is None else np.asarray(lam_grid, dtype=float)
    path = backward_elimination(regression, lam_grid, time_blocks(regression.n_steps, n_blocks))
    i_min, i_sel, threshold = one_standard_error(path)
    coef = regression.fit(path[i_sel].atoms, path[i_sel].lam_beta)
    return AtomSelection(path, i_min, i_sel, threshold, lam_grid, coef)
