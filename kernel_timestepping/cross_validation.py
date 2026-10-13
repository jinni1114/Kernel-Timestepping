"""Cross-validation helpers shared by the fits: held-out observation error and a grid search over one weight."""
from dataclasses import dataclass

import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def relative_l2(estimate, truth):
    """||estimate - truth|| / ||truth||."""
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth)) / np.linalg.norm(truth))


def heldout_rmse(fit, basis, obs, folds):
    """K-fold error of a state fit on held-out observations.

    fit: callable NodalObservations -> alpha (N_T, N_X), run on the observations outside each fold.
    Returns (RMS error per fold (n_folds,), RMS error pooled over all held-out observations).
    """
    residuals = []
    for fold in folds:
        u = np.asarray(basis.values(fit(obs.without(fold))))
        residuals.append(u[obs.n[fold], obs.k[fold]] - obs.y[fold])
    return np.array([rms(r) for r in residuals]), rms(np.concatenate(residuals))


@dataclass(frozen=True)
class GridSearch:
    """Cross-validation scores of the candidate values of one weight."""
    values: np.ndarray          # (n,)
    scores: np.ndarray          # (n,) pooled score per candidate
    fold_scores: np.ndarray     # (n, n_folds)

    @property
    def best_index(self):
        return int(np.argmin(self.scores))

    @property
    def best(self):
        return float(self.values[self.best_index])

    @property
    def best_score(self):
        return float(self.scores[self.best_index])


def grid_search(values, score):
    """Evaluate score(value) -> (fold scores, pooled score) on every candidate; the lowest pooled score wins."""
    results = [score(value) for value in values]
    return GridSearch(np.asarray(values, dtype=float), np.array([pooled for _, pooled in results]),
                      np.array([fold for fold, _ in results]))
