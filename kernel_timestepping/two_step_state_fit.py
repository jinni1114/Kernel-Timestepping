"""Step 1 of the two-step method: fit the state to the observations, all time columns together.

    min_alpha  sum_{(n,k) observed} (y_nk - (K alpha_n)_k)^2
               + lam_u sum_n alpha_n^T K alpha_n
               + lam_addot / dt^4 sum_{n=1}^{N_T-2} ||alpha_{n+1} - 2 alpha_n + alpha_{n-1}||_K^2

The columns share the spatial kernel and are coupled only through the last term, a prior on the discrete
acceleration u_tt. It smooths the state in time without shrinking the rate u_t, which is the target of the operator
fit. lam_addot = 0 fits every column on its own.

In whitened coordinates alpha_n = L^-T w_n (state_basis.py) both regularisers are Euclidean, so with w = (w_0, .., w_{N_T-1})
stacked the problem is one ridge regression,

    min_w ||M_aug w - y_aug||^2 + lam_u ||w||^2,
    M_aug = [M ; sqrt(lam_addot) / dt^2 (D_tt kron I)],   y_aug = [y ; 0],

M: one row per observation, (K L^-T)[k_i] in the block of column n_i; D_tt: second difference in time. Its normal
equations

    (M^T M + lam_addot / dt^4 (D_tt^T D_tt kron I) + lam_u I) w = M^T y

are block-pentadiagonal: M^T M is block-diagonal (one N_X x N_X block per column) and D_tt^T D_tt is pentadiagonal.
They are solved by a banded Cholesky factorisation (half-bandwidth 2 N_X), O(N_T N_X^3) instead of the
O((N_T N_X)^3) of a dense solve. `ridge_svd` solves the same problem through an SVD of M_aug, for checks.
"""
import numpy as np
from scipy.linalg import solveh_banded


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
    rows = basis.K[obs.k] @ basis.chol_inv.T                           # (n_obs, N_X): row i of M, within its column
    blocks = np.zeros((n_t, n_x, n_x))                                 # M^T M, block n
    np.add.at(blocks, obs.n, rows[:, :, None] * rows[:, None, :])
    rhs = np.zeros((n_t, n_x))                                         # M^T y, block n
    np.add.at(rhs, obs.n, obs.y[:, None] * rows)

    dtd = second_difference(n_t).T @ second_difference(n_t) * (lam_addot / grid.dt ** 4) if n_t > 2 else np.zeros((n_t, n_t))
    band = np.zeros((2 * n_x + 1, n_t * n_x))                          # lower band: band[i, j] = H[j + i, j]
    lower = np.tril_indices(n_x)
    for n in range(n_t):
        block = blocks[n] + (lam_u + dtd[n, n]) * np.eye(n_x)
        band[lower[0] - lower[1], n * n_x + lower[1]] = block[lower]
        for lag in (1, 2):                                             # time coupling: diagonal entries only
            if n + lag < n_t:
                band[lag * n_x, n * n_x + np.arange(n_x)] = dtd[n + lag, n]
    w = solveh_banded(band, rhs.ravel(), lower=True).reshape(n_t, n_x)
    return basis.unwhiten(w)
