"""Operator library of kernel modes (KMD): P_hat is a sum of atoms, each one a function in the RKHS of a scalar
kernel acting on one standardised feature of the state.

Features: g_i(s), i = 1..F, functions of the jet s = (u, u_x, ..., d^Q u) (e.g. u_xx, u * u_x). Each is standardised
with a centre c_i and scale sigma_i taken from a reference state: z_i = (g_i(s) - c_i) / sigma_i.

Atom m = (feature i(m), kernel k_m, inducing points zeta_m1..zeta_mJ) contributes

    P_hat_m(s) = sum_j k_m(z_{i(m)}, zeta_mj) beta_mj = phi_m(z_{i(m)})^T beta_m,
    ||P_hat_m||^2_{k_m} = beta_m^T K_m beta_m,    K_m = [k_m(zeta_mj, zeta_mj')].

The operator is P_hat(s) = sum_m P_hat_m(s) = Psi(s) b with b = (beta_1, ..., beta_M) stacked and Psi the design
matrix; the regulariser sum_m ||P_hat_m||^2 is b^T blockdiag(K_m) b. Kernels are JaxKernels kernels on scalars.
A rank-one kernel needs one inducing point: LinearKernel with zeta = 1 gives phi(z) = z, K_m = 1 (a term linear in
the feature), and ConstantKernel gives the constant atom phi = 1.
"""
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jaxkernels import ConstantKernel, GaussianRBFKernel, Kernel, LinearKernel, MaternKernel
from jaxkernels.kerneltools import vectorize_kfunc


class Feature(NamedTuple):
    """A scalar feature of the jet: fn maps s (..., Q + 1) = (u, u_x, ..., d^Q u) to (...)."""
    name: str
    fn: Callable


def default_features():
    """(u, u_x, u_xx, u_xxx, u u_x, u^2): needs jets up to the third derivative."""
    return (
        Feature("u", lambda s: s[..., 0]),
        Feature("u_x", lambda s: s[..., 1]),
        Feature("u_xx", lambda s: s[..., 2]),
        Feature("u_xxx", lambda s: s[..., 3]),
        Feature("u*u_x", lambda s: s[..., 0] * s[..., 1]),
        Feature("u^2", lambda s: s[..., 0] ** 2),
    )


def default_atom_kernels():
    """Unit-lengthscale kernels on a standardised feature: linear, Gaussian RBF, Matern 3/2."""
    return {"linear": LinearKernel(), "rbf": GaussianRBFKernel(lengthscale=1.0),
            "matern32": MaternKernel(p=1, lengthscale=1.0)}


@dataclass(frozen=True)
class Atom:
    """One kernel mode: kernel on the standardised feature number `feature` (None: the constant atom)."""
    name: str
    feature: Optional[int]
    kernel: Kernel
    inducing: np.ndarray        # (J,) inducing points, in standardised units
    prunable: bool = True

    @property
    def size(self):
        return len(self.inducing)

    def phi(self, z):
        """[k(z, zeta_j)]_j for z of any shape: (...) -> (..., J)."""
        z = jnp.asarray(z)
        return vectorize_kfunc(self.kernel)(z.reshape(-1), jnp.asarray(self.inducing)).reshape(z.shape + (self.size,))

    def gram(self):
        zeta = jnp.asarray(self.inducing)
        return np.asarray(vectorize_kfunc(self.kernel)(zeta, zeta))


@dataclass(frozen=True)
class AtomLibrary:
    """Features with their standardisation, and the atoms built on them (module docstring)."""
    features: tuple
    center: np.ndarray          # (F,)
    scale: np.ndarray           # (F,)
    atoms: tuple

    @property
    def names(self):
        return [atom.name for atom in self.atoms]

    @property
    def offsets(self):
        """beta_m = b[offsets[m]:offsets[m + 1]]."""
        return np.cumsum([0] + [atom.size for atom in self.atoms])

    @property
    def n_coef(self):
        return int(self.offsets[-1])

    def columns(self, atoms=None):
        """Positions in b of the coefficients of the atoms with indices `atoms` (default: all)."""
        offsets = self.offsets
        atoms = range(len(self.atoms)) if atoms is None else atoms
        return np.concatenate([np.arange(offsets[m], offsets[m + 1]) for m in atoms])

    def raw_features(self, jets):
        """g(s): jets (..., Q + 1) -> (..., F)."""
        return jnp.stack([feature.fn(jets) for feature in self.features], axis=-1)

    def standardize(self, jets):
        """z = (g(s) - c) / sigma: jets (..., Q + 1) -> (..., F)."""
        return (self.raw_features(jets) - self.center) / self.scale

    def design(self, jets):
        """Psi(s): jets (..., Q + 1) -> (..., n_coef), the blocks phi_m(z_{i(m)}) side by side."""
        z = self.standardize(jets)
        blocks = [atom.phi(z[..., 0 if atom.feature is None else atom.feature]) for atom in self.atoms]
        return jnp.concatenate(blocks, axis=-1)

    def evaluate(self, jets, coef):
        """P_hat(s) = Psi(s) b."""
        return self.design(jets) @ coef

    def penalty(self):
        """blockdiag(K_m), so that b^T penalty b = sum_m beta_m^T K_m beta_m."""
        offsets, out = self.offsets, np.zeros((self.n_coef, self.n_coef))
        for m, atom in enumerate(self.atoms):
            out[offsets[m]:offsets[m + 1], offsets[m]:offsets[m + 1]] = atom.gram()
        return out

    def restrict(self, atoms):
        """The library with only the atoms with indices `atoms` (same features and standardisation)."""
        return AtomLibrary(self.features, self.center, self.scale, tuple(self.atoms[m] for m in atoms))

    def split(self, coef):
        """{atom name: beta_m} for a stacked coefficient vector."""
        offsets = self.offsets
        return {atom.name: np.asarray(coef[offsets[m]:offsets[m + 1]]) for m, atom in enumerate(self.atoms)}

    def linear_coefficients(self, coef):
        """Coefficients in the original feature units of the atoms that are linear in their feature:
        ({feature name: a_i}, constant) with P_hat(s) = constant + sum_i a_i g_i(s) + (non-linear atoms).

        Undoes the standardisation: beta z = (beta / sigma) g - beta c / sigma."""
        offsets, slopes, constant = self.offsets, {}, 0.0
        for m, atom in enumerate(self.atoms):
            beta = np.asarray(coef[offsets[m]:offsets[m + 1]])
            if isinstance(atom.kernel, ConstantKernel):
                constant += float(beta[0] * atom.kernel.variance)
            elif isinstance(atom.kernel, LinearKernel):
                slope = float(beta[0] * atom.kernel.variance * atom.inducing[0]) / self.scale[atom.feature]
                name = self.features[atom.feature].name
                slopes[name] = slopes.get(name, 0.0) + slope
                constant -= slope * self.center[atom.feature]
        return slopes, constant


def inducing_points(kernel, z, n_inducing):
    """Inducing points for an atom kernel on the standardised reference values z: the single point 1 for the
    rank-one LinearKernel, otherwise n_inducing evenly spaced points over the range of z."""
    if isinstance(kernel, LinearKernel):
        return np.ones(1)
    return np.linspace(z.min(), z.max(), n_inducing)


def build_library(reference_jets, features=None, kernels=None, n_inducing=15, constant=True):
    """Library with one atom per (kernel, feature) pair, plus a constant atom that is never pruned.

    reference_jets: (..., Q + 1) jets of the reference state (the Step-1 fit); they fix the standardisation
    (mean, and standard deviation with ddof = 1, over all points) and the range of the inducing points.
    kernels: {name: JaxKernels kernel on scalars} (default: linear, RBF, Matern 3/2).
    """
    features = default_features() if features is None else tuple(features)
    kernels = default_atom_kernels() if kernels is None else dict(kernels)
    raw = np.stack([np.asarray(feature.fn(jnp.asarray(reference_jets))).ravel() for feature in features], axis=-1)
    center, scale = raw.mean(0), raw.std(0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    z = (raw - center) / scale

    atoms = [Atom("const", None, ConstantKernel(1.0), np.zeros(1), prunable=False)] if constant else []
    for kind, kernel in kernels.items():
        for i, feature in enumerate(features):
            atoms.append(Atom(f"{kind}[{feature.name}]", i, kernel, inducing_points(kernel, z[:, i], n_inducing)))
    return AtomLibrary(features, center, scale, tuple(atoms))
