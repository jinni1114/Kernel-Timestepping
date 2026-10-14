"""Spatial kernel representation of one time column of the state.

Every time column is a kernel expansion centred at the spatial nodes x_l,

    u_n(x) = sum_l k(x, x_l) alpha_{n, l},

so a state trajectory is an array alpha of shape (N_T, N_X) (row n = alpha_n). Its nodal values and its spatial
derivatives at the collocation nodes (spacetime_grid.py: the interior nodes, or every node if periodic) are linear in
alpha:

    u_n(x_k)                 = (K alpha_n)_k,     K_kl     = k(x_k, x_l)                       (N_X, N_X)
    d^q u_n / dx^q (x_k)     = (D_q alpha_n)_k,   (D_q)_kl = d^q/dx^q k(x, x_l) |_{x = x_k}    (N_C, N_X)

with N_C collocation nodes. The "jet" of a column is s(x_k; alpha_n) = (u, u_x, ..., d^Q u / dx^Q) at the collocation
nodes. D_q is built with the
JaxKernels functionals (kerneltools.derivative_op), so any scalar JaxKernels kernel that is Q times differentiable
in its first argument can be used.

Whitened coordinates: with K + nugget I = L L^T and alpha_n = L^-T w_n, the RKHS norm alpha_n^T K alpha_n is
(up to the nugget) ||w_n||^2, which turns the state regularisers into plain ridge penalties.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxkernels import Kernel
from jaxkernels.kerneltools import derivative_op, eval_k, make_block

from .spacetime_grid import SpaceTimeGrid


def collocation_matrix(kernel, x_eval, x_centers, order=0):
    """[d^order/dx^order k(x, c_l) at x = x_eval_k]_kl for scalar points; shape (len(x_eval), len(x_centers))."""
    block = make_block(kernel, derivative_op(order), eval_k)
    return np.asarray(block(jnp.asarray(x_eval, dtype=float), jnp.asarray(x_centers, dtype=float)))


@dataclass(frozen=True)
class StateBasis:
    """Kernel basis of a state column on the nodes x, with its collocation matrices (module docstring)."""
    kernel: Kernel
    x: np.ndarray           # (N_X,) nodes = kernel centres
    K: np.ndarray           # (N_X, N_X)
    D: np.ndarray           # (Q + 1, N_C, N_X): D[q] gives the q-th x-derivative at the collocation nodes
    chol: np.ndarray        # L, lower triangular, K + nugget I = L L^T
    chol_inv: np.ndarray    # L^-1
    collocation: np.ndarray  # (N_C,) indices of the collocation nodes in x

    @classmethod
    def build(cls, kernel, grid, max_order=3, nugget=1e-8):
        """grid: a SpaceTimeGrid (its nodes and collocation nodes), or an array of nodes with fixed boundary values
        at both ends (collocation at x[1:-1])."""
        if isinstance(grid, SpaceTimeGrid):
            x, collocation = grid.x, grid.collocation
        else:
            x = np.asarray(grid, dtype=float)
            collocation = np.arange(1, len(x) - 1)
        K = collocation_matrix(kernel, x, x)
        D = np.stack([collocation_matrix(kernel, x[collocation], x, q) for q in range(max_order + 1)])
        chol = np.linalg.cholesky(K + nugget * np.eye(len(x)))
        return cls(kernel, x, K, D, chol, np.linalg.inv(chol), collocation)

    @property
    def n_x(self):
        return len(self.x)

    @property
    def max_order(self):
        return self.D.shape[0] - 1

    def matrix(self, x_eval, order=0):
        """Collocation matrix of the order-th x-derivative at arbitrary points: (len(x_eval), N_X)."""
        return collocation_matrix(self.kernel, x_eval, self.x, order)

    def values(self, alpha):
        """Nodal values K alpha_n for alpha (..., N_X) -> (..., N_X)."""
        return alpha @ self.K                                   # K is symmetric

    def jets(self, alpha):
        """s(x_k; alpha_n) = (u, u_x, ..., d^Q u) at the collocation nodes: alpha (..., N_X) -> (..., N_C, Q + 1)."""
        return jnp.stack([alpha @ D_q.T for D_q in self.D], axis=-1)

    def interpolate(self, u_nodes):
        """Coefficients alpha with K alpha = u_nodes (rows of u_nodes are columns of the state)."""
        return np.linalg.solve(self.K, np.asarray(u_nodes).T).T

    def whiten(self, alpha):
        """w = L^T alpha (row-wise), so that alpha^T K alpha ~ ||w||^2."""
        return alpha @ self.chol

    def unwhiten(self, w):
        """alpha = L^-T w (row-wise)."""
        return w @ self.chol_inv
