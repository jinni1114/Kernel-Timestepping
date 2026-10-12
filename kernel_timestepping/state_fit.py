"""Step 1 of the two-step method: fit the state to the observations, all time columns together.

    min_alpha  sum_{(n,k) observed} (y_nk - (K alpha_n)_k)^2
               + lam_u sum_n alpha_n^T K alpha_n
               + lam_addot / dt^4 sum_{n=1}^{N_T-2} ||alpha_{n+1} - 2 alpha_n + alpha_{n-1}||_K^2

The columns share the spatial kernel and are coupled only through the last term, a prior on the discrete
acceleration u_tt. It smooths the state in time without shrinking the rate u_t, which is the target of the operator
fit. lam_addot = 0 fits every column on its own.

In whitened coordinates alpha_n = L^-T w_n (state.py) both regularisers are Euclidean, so with w = (w_0, .., w_{N_T-1})
stacked the problem is one ridge regression, solved through an SVD:

    min_w ||M_aug w - y_aug||^2 + lam_u ||w||^2,
    M_aug = [M ; sqrt(lam_addot) / dt^2 (D_tt kron I)],   y_aug = [y ; 0],

M: one row per observation, (K L^-T)[k_i] in the block of column n_i; D_tt: second difference in time.
"""
import numpy as np


def second_difference(n):
    """(n - 2, n) matrix with rows (1, -2, 1): (D w)_n = w_n - 2 w_{n+1} + w_{n+2}."""
    return (np.eye(n, k=2) - 2 * np.eye(n, k=1) + np.eye(n))[:-2]


def ridge_svd(M, y, reg):
    """argmin_w ||M w - y||^2 + reg ||w||^2 through the SVD of M."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt.T @ ((s / (s ** 2 + reg)) * (U.T @ y))


def fit_state(basis, grid, obs, lam_u=1e-6, lam_addot=0.0):
    """Coefficients alpha (N_T, N_X) of the regularised state fit (module docstring)."""
    n_t, n_x = grid.n_t, grid.n_x
    M = np.zeros((len(obs), n_t * n_x))
    block_columns = obs.n[:, None] * n_x + np.arange(n_x)[None, :]
    M[np.arange(len(obs))[:, None], block_columns] = basis.K[obs.k] @ basis.chol_inv.T
    time_rows = np.kron(second_difference(n_t), np.eye(n_x)) / grid.dt ** 2
    M_aug = np.vstack([M, np.sqrt(lam_addot) * time_rows])
    y_aug = np.concatenate([obs.y, np.zeros(len(time_rows))])
    w = ridge_svd(M_aug, y_aug, lam_u).reshape(n_t, n_x)
    return basis.unwhiten(w)
