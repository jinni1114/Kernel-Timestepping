"""KdV walkthrough (Raissi two-soliton), 5 % random interior observations: 2-step soft/hard, 1-step soft/hard.

Script version of timestepping_kdv (Colab notebook). Cells are marked "# %%" (runs cell by cell in VS Code / Spyder /
Jupytext). Before running outside Colab:
    git clone https://github.com/jinni1114/Kernel-Timestepping.git
    pip install "./Kernel-Timestepping[examples]"
and set KT_REPO below (or the environment variable KT_REPO) to the clone.
"""

# %%
# Clone the repository and install the package (no runtime restart needed)
# git clone -q https://github.com/jinni1114/Kernel-Timestepping.git /content/Kernel-Timestepping    (shell command: run once before this script)
# pip install -q "/content/Kernel-Timestepping[examples]"    (shell command: run once before this script)

import sys
import os
KT_REPO = os.environ.get("KT_REPO", "/content/Kernel-Timestepping")       # the cloned repository
sys.path[:0] = [KT_REPO, os.path.join(KT_REPO, "experiments")]             # problems/, plot_style

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxkernels import PeriodicKernel

import kernel_timestepping as kt
from problems.periodic_reference import PeriodicReferenceSolution, kdv, solve_periodic
import plot_style as ph
ph.use_style()

# %% [markdown]
# KdV data

# %%
A, B = 6.0, 1.0                                            # the equation to learn: u_t = -A u u_x - B u_xxx
X_MIN, PERIOD = -30.0, 60.0                                # x in [-30, 30), periodic
EQUATION = kdv(a=A, b=B)
N_X, DT, T_TRAIN, T_ROLL = 128, 0.2, 10.0, 20.0           # collocation grid: 128 nodes per time slice (resolves the solitons)
OBSERVED = 0.05                                        # fraction of the nodes after the initial column


def soliton(x, c, x0):
    """KdV soliton of speed c centred at x0 (periodic copy): c/2 sech^2(sqrt(c)/2 (x - x0))."""
    return c / 2 / np.cosh(np.sqrt(c) / 2 * (((x - x0) - X_MIN) % PERIOD + X_MIN)) ** 2


class Reference:
    """Spectral reference solution on [X_MIN, X_MIN + PERIOD): the solver works on [0, PERIOD), so x is shifted."""

    def __init__(self, u0, t_final=T_ROLL):
        t, x, u = solve_periodic(EQUATION, lambda x: u0(x + X_MIN), PERIOD, n=512, dt=1e-3, t_final=t_final,
                                 save_every=50)                       # ETDRK4, 512 modes, saved every 0.05
        self.series = PeriodicReferenceSolution(EQUATION, t, x, u, PERIOD)

    def u(self, t, x, grid=True):
        return self.series.u(t, np.asarray(x) - X_MIN, grid)

    def u_t(self, t, x, grid=True):
        return self.series.u_t(t, np.asarray(x) - X_MIN, grid)

    def derivative(self, t, x, order=0, grid=True):
        return self.series.derivative(t, np.asarray(x) - X_MIN, order, grid)

    def jets(self, t, x, max_order=3):
        return self.series.jets(t, np.asarray(x) - X_MIN, max_order)


u0_train = lambda x: soliton(x, 1.0, -20.0) + soliton(x, 0.5, 5.0)    # amplitudes 0.5 and 0.25, speeds 1 and 0.5
u0_test = lambda x: soliton(x, 0.8, -15.0) + soliton(x, 0.4, 10.0)    # never shown to the fit (section 6)
truth, truth_test = Reference(u0_train), Reference(u0_test)

n_t = int(round(T_TRAIN / DT)) + 1
grid = kt.SpaceTimeGrid(x=kt.periodic_nodes(N_X, PERIOD, x0=X_MIN), dt=DT, n_t=n_t, period=PERIOD)
obs = kt.sample_nodal_observations(grid, lambda t, x: truth.u(t, x, grid=False),
                                   n_interior=int(OBSERVED * N_X * (n_t - 1)), seed=0)

u_true = truth.u(grid.t, grid.x)                           # (51, 128) truth at the nodes
jets_true = truth.jets(grid.t, grid.x_int, max_order=3)    # (51, 128, 4) true (u, u_x, u_xx, u_xxx) at the nodes
print(f"{grid.n_t} x {grid.n_x} = {grid.n_t * grid.n_x} nodes, {len(obs)} observed "
      f"({len(obs) / (grid.n_t * grid.n_x):.0%}): {(obs.n == 0).sum()} initial, {obs.is_interior(grid).sum()} later")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", width_ratios=[1.25, 1])

t_fine, x_fine = np.linspace(0, T_TRAIN, 201), np.linspace(X_MIN, X_MIN + PERIOD, 601)
limit = np.abs(u_true).max()
mesh = ph.field_map(axes[0], t_fine, x_fine, truth.u(t_fine, x_fine), limit, "Reference solution and what the fit sees")
t_nodes, x_nodes = np.meshgrid(grid.t, grid.x, indexing="ij")
axes[0].scatter(t_nodes, x_nodes, s=1, color=ph.MUTED, linewidths=0, label="grid node")
axes[0].scatter(grid.t[obs.n], grid.x[obs.k], s=6, color=ph.INK, edgecolors=ph.SURFACE, linewidths=0.3,
                label="observed node", clip_on=False, zorder=3)
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.colorbar(mesh, ax=axes[0], pad=0.02, label="u")

snapshot_colors = ["#86b6ef", "#2a78d6", "#104281"]         # one hue, light to dark: earlier to later
for time, color in zip([0.0, 5.0, 10.0], snapshot_colors):
    axes[1].plot(x_fine, truth.u(time, x_fine)[0], color=color, label=f"t = {time:g}")
axes[1].axhline(0, color=ph.AXIS, lw=0.8, zorder=0)
axes[1].set(title="Snapshots of the reference solution", xlabel="x", ylabel="u")
axes[1].legend()
plt.show()

# %% [markdown]
# 2-step soft constraint

# %% [markdown]
# Pick state kernel and evaluate feature matrices

# %%
# Periodic Gaussian kernel. Lengthscale: 0.75 x the space lengthscale of a GP fit to the observations (as for Burgers),
# reduced if needed so that cond(K) <= 1e12 (a smooth periodic kernel at the GP lengthscale is numerically singular)
ell_x = kt.state_lengthscale(grid, obs, max_cond=1e12)
basis = kt.StateBasis.build(PeriodicKernel(period=PERIOD, lengthscale=ell_x), grid, max_order=3)
print(f"lengthscale {ell_x:.4f} (GP fit x 0.75 would be {kt.state_lengthscale(grid, obs):.4f});  K {basis.K.shape}, "
      f"D {basis.D.shape} (derivative order, node, centre);  cond(K) = {np.linalg.cond(basis.K):.1e}")

# %% [markdown]
# state fit

# %%
LAM_U = 1e-6
lam_addot_grid = np.r_[0, np.logspace(-12, -1, 23)]
folds = kt.interior_folds(grid, obs, n_folds=5, seed=0)       # held out: observations outside the initial column


def state_cv(lam_addot):
    return kt.heldout_rmse(lambda train: kt.fit_state(basis, grid, train, LAM_U, lam_addot), basis, obs, folds)


cv_state = kt.grid_search(lam_addot_grid, state_cv)
alpha_bar = kt.fit_state(basis, grid, obs, LAM_U, cv_state.best)             # (51, 128), row n = alpha_n
alpha_columnwise = kt.fit_state(basis, grid, obs, LAM_U, 0.0)                # for comparison: no time prior

jets_bar = np.asarray(basis.jets(alpha_bar))
print(f"chosen lam_addot = {cv_state.best:.1e}: held-out RMS error {cv_state.best_score:.2e} "
      f"(columns on their own: {cv_state.scores[0]:.2e})")
print("relative L2 error of the fitted state at the nodes: "
      + ",  ".join(f"{name} {kt.relative_l2(jets_bar[..., q], jets_true[..., q]):.3f}"
                   for q, name in enumerate(["u", "u_x", "u_xx", "u_xxx"])))

# Visualize state fit
fig = plt.figure(figsize=(13, 5.2), layout="constrained")
panels = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1])

ax = fig.add_subplot(panels[:, 0])
positive = cv_state.values[1:]
positions = np.r_[positive[0] / 10, positive]                # lam_addot = 0 is drawn one decade left of the grid
ax.plot(positions, cv_state.fold_scores, color=ph.NEUTRAL, lw=1)
ax.plot([], [], color=ph.NEUTRAL, lw=1, label="single fold")
ax.plot(positions, cv_state.scores, color=ph.BLUE, label="pooled over 5 folds")
ph.marker(ax, positions[cv_state.best_index], cv_state.best_score, ph.BLUE, ms=9, ls="none",
          label=f"chosen weight: {cv_state.best:.0e}")
ax.set(xscale="log", yscale="log", xlabel="weight of the time prior (leftmost point: 0)",
       ylabel="held-out RMS error", title="Cross-validation of the time prior")
ax.set_xticks([positions[0], 1e-10, 1e-7, 1e-4, 1e-1], ["0", "1e-10", "1e-7", "1e-4", "1e-1"])
ax.grid(True, axis="y")
ax.legend(loc="upper center")

derivative_names = {0: "u", 3: "u_xxx"}                      # KdV's linear term is u_xxx (Burgers showed u_xx)
for row, order in enumerate(derivative_names):
    B_fine = basis.matrix(x_fine, order)                     # (601, 128): basis functions (or derivatives) on a fine grid
    for col, n in enumerate([10, 25, 50]):                   # t = 2, 5, 10
        ax = fig.add_subplot(panels[row, col + 1])
        exact = truth.derivative(grid.t[n], x_fine, order)[0]
        ax.plot(x_fine, exact, color=ph.NEUTRAL, lw=5, label="truth")
        ax.plot(x_fine, B_fine @ alpha_columnwise[n], color=ph.ORANGE, lw=1.5, label="columns on their own")
        ax.plot(x_fine, B_fine @ alpha_bar[n], color=ph.BLUE, label="with the time prior")
        if order == 0:
            here = obs.n == n
            ax.plot(grid.x[obs.k[here]], obs.y[here], "o", color=ph.INK, mec=ph.SURFACE, mew=1.0, ms=5,
                    label="observation", zorder=4)
            ax.set_title(f"t = {grid.t[n]:.1f}  ({here.sum()} observations)")
        span = exact.max() - exact.min()
        ax.set(ylim=(exact.min() - 0.35 * span, exact.max() + 0.35 * span), ylabel=derivative_names[order])
        if row == 1:
            ax.set_xlabel("x")
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=7)
plt.show()

x_plot = np.linspace(X_MIN, X_MIN + PERIOD, 481)
names = ["u", "u_x", "u_xx", "u_xxx"]
fig, axes = plt.subplots(len(names), 2, figsize=(10, 3.0 * len(names)), layout="constrained", sharex=True, sharey=True)
for q, name in enumerate(names):
    exact = truth.derivative(grid.t, x_plot, q)                  # (51, 481)
    fitted = alpha_bar @ basis.matrix(x_plot, q).T               # (51, 481): d^q/dx^q of the Step-1 state
    limit = np.abs(exact).max()
    ph.field_map(axes[q, 0], grid.t, x_plot, exact, limit, f"{name}: truth")
    mesh = ph.field_map(axes[q, 1], grid.t, x_plot, fitted, limit,
                        f"{name}: ours (Step 1)   rel. L2 error {kt.relative_l2(fitted, exact):.2e}")
    fig.colorbar(mesh, ax=axes[q, :], pad=0.01, label=name, extend="both")
    if q == 0:
        for ax in axes[0]:
            ax.scatter(grid.t[obs.n], grid.x[obs.k], s=2, color=ph.INK, linewidths=0)
for ax in axes.ravel():
    ax.set(xlabel="t", ylabel="x")
    ax.label_outer()
plt.show()

# %% [markdown]
# Operator KMD library

# %%
# One atom per (kernel, feature): linear (1 coefficient), Gaussian RBF and Matérn 3/2 (15 coefficients each), on the
# features (u, u_x, u_xx, u_xxx, u u_x, u^2) standardised on the Step-1 state; plus a constant atom that is never pruned
library = kt.build_library(basis.jets(alpha_bar), n_inducing=15)

print(f"{len(library.atoms)} atoms, {library.n_coef} coefficients\n")
print(f"{'feature':8s} {'mean':>10s} {'std':>10s}   coefficients per atom")
for i, feature in enumerate(library.features):
    on_feature = [atom for atom in library.atoms if atom.feature == i]
    print(f"{feature.name:8s} {library.center[i]:10.3g} {library.scale[i]:10.3g}   "
          + ", ".join(f"{atom.name.split('[')[0]} {atom.size}" for atom in on_feature))
print(f"{'(none)':8s} {'':>10s} {'':>10s}   const 1")

# ---- check 1: the u_xxx collocation matrix D_3 against a centred difference of the u_xx matrix (relative)
h = 1e-5
D3_fd = (basis.matrix(grid.x_int + h, 2) - basis.matrix(grid.x_int - h, 2)) / (2 * h)
print(f"D_3 vs finite difference of D_2 (relative): {np.abs(basis.D[3] - D3_fd).max() / np.abs(basis.D[3]).max():.1e}")

# ---- check 2: the true operator -A u u_x - B u_xxx is in the library: const + linear[u_xxx] + linear[u*u_x]
# (an atom acts on z = (g - c) / sigma, so g = sigma z + c gives the coefficients below)
names = [f.name for f in library.features]
i3, i4 = names.index("u_xxx"), names.index("u*u_x")
true_atoms = [library.names.index("const"), library.names.index("linear[u_xxx]"), library.names.index("linear[u*u_x]")]
coef_true = np.array([-B * library.center[i3] - A * library.center[i4], -B * library.scale[i3], -A * library.scale[i4]])
P_library = np.asarray(library.restrict(true_atoms).evaluate(jnp.asarray(jets_true), jnp.asarray(coef_true)))
P_exact = -A * jets_true[..., 0] * jets_true[..., 1] - B * jets_true[..., 3]
print(f"true operator through the library vs -A u u_x - B u_xxx (relative): "
      f"{np.abs(P_library - P_exact).max() / np.abs(P_exact).max():.1e}")
print(f"-A u u_x - B u_xxx vs the reference u_t at the nodes (relative): "
      f"{np.abs(P_exact - truth.u_t(grid.t, grid.x)).max() / np.abs(P_exact).max():.1e}")

# ---- visual check: each feature, standardised, at the Step-1 state (histogram over all nodes and columns) and at the
# TRUE state (grey outline); ticks: the 15 inducing points of its RBF / Matern 3/2 atoms
Z_bar = np.asarray(library.standardize(jnp.asarray(jets_bar))).reshape(-1, len(names))
Z_true = np.asarray(library.standardize(jnp.asarray(jets_true))).reshape(-1, len(names))
fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), layout="constrained")
for i, (ax, name) in enumerate(zip(axes, names)):
    lo, hi = np.percentile(np.r_[Z_bar[:, i], Z_true[:, i]], [0.5, 99.5])
    bins = np.linspace(lo, hi, 40)
    ax.hist(Z_bar[:, i], bins=bins, color=ph.BLUE, alpha=0.7, label="Step 1 state")
    ax.hist(Z_true[:, i], bins=bins, histtype="step", color=ph.MUTED, lw=1.5, label="true state")
    rbf = library.atoms[library.names.index(f"rbf[{name}]")]
    for zeta in rbf.inducing:
        ax.axvline(zeta, ymax=0.08, color=ph.INK, lw=1)
    ax.set(title=f"{name}  (c = {library.center[i]:.2g}, sigma = {library.scale[i]:.2g})",
           xlabel="standardised feature z", yticks=[], yscale="log")
axes[0].legend(fontsize=8)
fig.suptitle(f"Library features over the {Z_bar.shape[0]} nodes x columns; ticks = inducing points of the RBF / "
             f"Matern 3/2 atoms", x=0.01, ha="left")
plt.show()

# %% [markdown]
# Operator fit

# %%
# =====================================================================================
# 2-step method, Step 2: operator fit on the frozen Step 1 state (soft constraint)
# =====================================================================================
# Operator: the KMD library (Cell 8), P_hat(s) = Psi(s) b = sum over active atoms m of phi_m(z_{i(m)})^T beta_m,
#   s = (u, u_x, u_xx, u_xxx).
# Dynamic residual, theta = 1/2 (Crank-Nicolson), every node collocated (periodic: no boundary rows):
#   e_n = K (alpha_bar_{n+1} - alpha_bar_n) - dt [theta r(alpha_bar_{n+1}; b) + (1 - theta) r(alpha_bar_n; b)] = d_n - A_n b,
#   d_n = K alpha_bar_{n+1} - K alpha_bar_n,   A_n = dt [theta Psi(s_{n+1}) + (1 - theta) Psi(s_n)].
# Loss (state frozen at alpha_bar, active atom set A):
#   J(b) = lam_dyn sum_n ||e_n||^2 + lam_beta sum_{m in A} beta_m^T K_m beta_m,  lam_dyn = 1, closed-form minimiser.
# lam_beta by 5-fold CV over contiguous blocks of time steps (score: RMS of e on the held-out steps).
# Atom selection: backward elimination (drop, at each step, the atom whose removal gives the lowest CV, lam_beta
#   re-chosen for every candidate set; the constant atom is always kept), then the one-standard-error rule: the set
#   with the fewest atoms whose CV <= CV_min + SE_min.
THETA = 0.5
regression = kt.OperatorRegression.from_state(basis, library, alpha_bar, grid.dt, theta=THETA)
selection = kt.select_atoms(regression, lam_grid=np.logspace(-12, 2, 15), n_blocks=5)
active = library.restrict(selection.atoms)                  # the library with the selected atoms only
coef_soft = selection.coef                                  # their coefficients (standardised units)
path, sel = selection.path, selection.selected


def describe(library, coef):
    """The fitted operator in the original units of the features."""
    slopes, constant = library.linear_coefficients(coef)
    text = f"{constant:+.2e} " + " ".join(f"{value:+.4g} {name}" for name, value in slopes.items())
    kernel_atoms = [atom.name for atom in library.atoms if atom.size > 1]
    return text + (" + " + " + ".join(kernel_atoms) if kernel_atoms else "")


n_atoms = lambda step: sum(library.atoms[m].prunable for m in step.atoms)   # number of non-constant atoms
best = path[selection.i_min]
print(f"backward elimination: min CV {best.cv:.3e} +- {best.se:.1e} at {n_atoms(best)} atoms; 1-SE rule -> "
      f"{n_atoms(sel)} atoms (CV {sel.cv:.3e}, lam_beta {selection.lam_beta:.0e}): "
      + ", ".join(name for name in active.names if name != "const"))
print(f"learned  P = {describe(active, coef_soft)}")
print(f"true     P = {-A:+.4g} u*u_x {-B:+.4g} u_xxx")

# ---- truth-based diagnostics (NOT used for the selection): P_hat at the TRUE features on a 100 x 100 (t, x) grid
#      inside the training window, against the true u_t;  nrmse = sqrt(mean (P_hat - u_t)^2 / mean u_t^2)
g_t = np.linspace(0, T_TRAIN, 104)[2:-2]                                # 100 times inside (0, T_TRAIN)
g_x = X_MIN + PERIOD * np.arange(100) / 100                             # 100 points over the period
jets_eval = truth.jets(g_t, g_x, max_order=3).reshape(-1, 4)            # (10000, 4) true (u, u_x, u_xx, u_xxx), t-major
ut_eval = truth.u_t(g_t, g_x).ravel()
nrmse = lambda pred, target=ut_eval: np.sqrt(np.mean((pred - target) ** 2) / np.mean(target ** 2))
P_true = lambda jets: -A * jets[..., 0] * jets[..., 1] - B * jets[..., 3]

h = 1e-4                                                                # independent u_t: centred difference in t
ut_fd = ((truth.u(g_t + h, g_x) - truth.u(g_t - h, g_x)) / (2 * h)).ravel()
print(f"check: true operator vs a time difference of the reference (spline in t), nrmse {nrmse(P_true(jets_eval), ut_fd):.1e}")


def operator_at_true_features(atoms, coef):
    return np.asarray(library.restrict(atoms).evaluate(jnp.asarray(jets_eval), jnp.asarray(coef)))


op_path = [nrmse(operator_at_true_features(st.atoms, regression.fit(st.atoms, st.lam_beta))) for st in path]
P_eval = operator_at_true_features(selection.atoms, coef_soft)

# ---- floor of the dynamic residual: e at the TRUE state with the TRUE operator, relative to K(alpha_{n+1} - alpha_n)
#      (the Crank-Nicolson truncation error, about -dt^3/12 u_ttt per step)
d_tr = u_true[1:] - u_true[:-1]
P_tr = P_true(jets_true)
e_tr = d_tr - DT * (THETA * P_tr[1:] + (1 - THETA) * P_tr[:-1])
e_soft = regression.residual(selection.atoms, coef_soft)
print(f"e relative to K(alpha_(n+1) - alpha_n): at the truth {np.linalg.norm(e_tr) / np.linalg.norm(d_tr):.3f}, "
      f"Step 1 state with the selected operator {np.linalg.norm(e_soft) / np.linalg.norm(regression.d):.3f}")
print(f"operator nrmse at the true features: selected set {nrmse(P_eval):.3e}")

# =====================================================================================
# Visual check on the 100 x 100 grid, P_hat evaluated at the TRUE features.
# Columns: true u_t | ours P_hat | ours P_hat - u_t | CV curve for lam_beta (selected set).
# =====================================================================================
nt_e, nx_e = len(g_t), len(g_x)
lim = np.abs(ut_eval).max()
err = P_eval - ut_eval
elim = np.percentile(np.abs(err), 99.5)
fig, ax = plt.subplots(1, 4, figsize=(22, 4.6), layout="constrained")
ph.field_map(ax[0], g_t, g_x, ut_eval.reshape(nt_e, nx_e), lim, "true u_t")
im = ph.field_map(ax[1], g_t, g_x, P_eval.reshape(nt_e, nx_e), lim, f"P_hat(s_true): ours\nnrmse {nrmse(P_eval):.3e}")
fig.colorbar(im, ax=ax[1], fraction=.046, pad=.02, label="u_t")
im2 = ax[2].pcolormesh(ph.cell_edges(g_t), ph.cell_edges(g_x), err.reshape(nt_e, nx_e).T, cmap="PuOr_r",
                       norm=ph.TwoSlopeNorm(0, -elim, elim), shading="flat", rasterized=True)
ax[2].set(title=f"P_hat - u_t: ours\nnrmse {nrmse(P_eval):.3e}", xlabel="t", ylabel="x")
fig.colorbar(im2, ax=ax[2], fraction=.046, pad=.02, extend="both", label="P_hat - u_t")
a = ax[3]                                                               # CV for lam_beta, selected set
for j in range(sel.cv_blocks.shape[1]):
    a.loglog(selection.lam_grid, sel.cv_blocks[:, j], "-", c=ph.BLUE, lw=.8, alpha=.35,
             label="single block of steps" if j == 0 else None)
a.loglog(selection.lam_grid, sel.cv_curve, "o-", c=ph.BLUE, lw=2.2, ms=5, label="pooled")
a.axvline(selection.lam_beta, c=ph.INK, lw=1, ls=":", label=f"chosen {selection.lam_beta:.0e}")
a.set(title=f"2-step soft: time-block CV for lam_beta (selected set, {n_atoms(sel)} atoms)",
      xlabel="lam_beta  (lam_dyn = 1)", ylabel="held-out RMS of e")
a.legend(fontsize=8)
plt.show()

# =====================================================================================
# Visual check of the atom selection.
# (a) pooled CV +- 1 SE (5 time blocks) along the backward-elimination path; dashed = CV_min + SE_min.
# (b) atoms in the set at each step (orange = selected set), and the atom dropped at that step (x).
# (c) operator nrmse at the true features along the path (diagnostic only, NOT used for the selection).
# (d) the learned additive component of each feature (sum over selected atoms on it of phi_m(z) beta_m) against the
#     feature value over the Step 1 points, and the true component (-A u u_x, -B u_xxx, 0 otherwise), both centred
#     (the constant atom absorbs the means).
# =====================================================================================
nonconst = [m for m, atom in enumerate(library.atoms) if atom.prunable]
na = np.array([n_atoms(st) for st in path])
fig = plt.figure(figsize=(26, 10), layout="constrained")
gs = fig.add_gridspec(2, 6)
a = fig.add_subplot(gs[0, 0:2])
a.errorbar(na, [st.cv for st in path], yerr=[st.se for st in path], fmt="o-", c=ph.BLUE, ms=4, capsize=3)
a.axhline(selection.threshold, c=ph.MUTED, ls="--", lw=1, label="CV_min + SE_min")
a.plot(na[selection.i_min], best.cv, "s", ms=11, mfc="none", mec=ph.INK, mew=1.5,
       label=f"min CV ({na[selection.i_min]} atoms)")
a.plot(na[selection.i_sel], sel.cv, "o", ms=13, mfc="none", mec=ph.ORANGE, mew=2,
       label=f"1-SE choice ({na[selection.i_sel]} atoms)")
a.invert_xaxis(); a.set_yscale("log"); a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
a.set(title="(a) backward elimination: held-out CV +- 1 SE", xlabel="number of atoms (path runs left to right)",
      ylabel="RMS of e (held-out steps)")
a.legend(fontsize=8)
a = fig.add_subplot(gs[1, 0:2])
a.semilogy(na, op_path, "o-", c=ph.BLUE, ms=4, label="ours 2-step soft")
a.axvline(na[selection.i_sel], c=ph.ORANGE, lw=1.5, ls=":", label="1-SE choice")
a.invert_xaxis(); a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
a.set(title="(c) operator nrmse along the path (diagnostic only)", xlabel="number of atoms", ylabel="nrmse")
a.legend(fontsize=8)
a = fig.add_subplot(gs[:, 2])
for c_, st in enumerate(path):
    for row, m in enumerate(nonconst):
        if m in st.atoms:
            a.add_patch(plt.Rectangle((c_ - .45, row - .45), .9, .9, lw=0,
                                      color=ph.ORANGE if c_ == selection.i_sel else ph.BLUE,
                                      alpha=.85 if c_ == selection.i_sel else .35))
        if st.dropped == m:
            a.plot(c_, row, "x", c=ph.INK, ms=7, mew=1.5)
a.set_xlim(-.6, len(path) - .4); a.set_ylim(len(nonconst) - .5, -.5)
a.set_yticks(range(len(nonconst)), [library.atoms[m].name for m in nonconst], fontsize=8)
a.set_xticks(range(len(path)), na, fontsize=7)
a.set(title="(b) atoms in the set (x = dropped next)", xlabel="number of atoms")
TRUE_COMP = {"u_xxx": -B, "u*u_x": -A}                                  # true operator: -A u u_x - B u_xxx
G_bar = np.asarray(library.raw_features(jnp.asarray(jets_bar))).reshape(-1, len(library.features))
beta_soft = active.split(coef_soft)
for i, feature in enumerate(library.features):
    a = fig.add_subplot(gs[i // 3, 3 + i % 3])
    gg = np.linspace(G_bar[:, i].min(), G_bar[:, i].max(), 200)
    z = (gg - library.center[i]) / library.scale[i]
    on = [atom for atom in active.atoms if atom.feature == i]
    comp = sum((np.asarray(atom.phi(jnp.asarray(z))) @ beta_soft[atom.name] for atom in on), np.zeros_like(gg))
    tr = TRUE_COMP.get(feature.name, 0.0) * gg
    a.plot(gg, tr - tr.mean(), c=ph.MUTED, lw=3, alpha=.6, label="true component")
    a.plot(gg, comp - comp.mean(), c=ph.BLUE, lw=1.8,
           label="learned: " + (", ".join(atom.name.split("[")[0] for atom in on) if on else "none"))
    a.set(title=f"(d) component of {feature.name}", xlabel=feature.name)
    a.legend(fontsize=7)
fig.suptitle("Operator KMD on the 2-step soft fit: backward elimination + 1-SE rule", x=0.01, ha="left", fontsize=12)
plt.show()

# %% [markdown]
# 2-step hard constraint

# %%
# =====================================================================================
# 2-step method, Step 2 HARD: operator fitted THROUGH the Crank-Nicolson rollout (state frozen at alpha_bar)
# =====================================================================================
# Operator: the KMD operator with the atoms selected by 2-step soft (fixed here); only their coefficients b_A are
#   unknowns:  P_hat(s) = sum_{m in A} phi_m(z_{i(m)})^T beta_m,   s = (u, u_x, u_xx, u_xxx).
# Step map alpha_{n+1} = Phi(alpha_n; b) (kt.ThetaStepper): Newton on the dynamic residual
#   e(alpha_{n+1}; b) = K (alpha_{n+1} - alpha_n) - dt [theta r(alpha_{n+1}; b) + (1 - theta) r(alpha_n; b)] = 0,
#   r(alpha; b)_k = P_hat(s(x_k; alpha)) at every node (periodic: no boundary rows),  theta = 1/2,
#   6 Newton iterations from alpha_n with J = K - dt theta dr/dalpha (jax.jacfwd); (1 - theta) r(alpha_n; b) computed once.
# Rollout: alpha_0 = alpha_bar_0,  alpha_n = Phi^n(alpha_bar_0; b), n = 1..N_T-1 (no data used after t = 0).
# Loss:  J(b) = sum_{n >= 1} || K alpha_bar_n - K Phi^n(alpha_bar_0; b) ||^2 + lam_beta sum_{m in A} beta_m^T K_m beta_m,
#   lam_beta = LAM_BETA_2H (fixed). Nonlinear in b through the rollout: trust-region Gauss-Newton (scipy least_squares)
#   with the exact Jacobian (jax.jacfwd through the Newton steps)  ->  kt.fit_two_step_operator_hard.
NEWTON_ITERS = 6
LAM_BETA_2H = 1e-6
MAX_NFEV = 300

stepper = kt.ThetaStepper(basis, active.evaluate, grid.dt, theta=THETA, newton_iters=NEWTON_ITERS)   # ours (KMD operator)
true_stepper = kt.ThetaStepper(basis, lambda jets, _: P_true(jets), grid.dt, theta=THETA, newton_iters=NEWTON_ITERS)

# warm start from the soft fit; b = 0 as a check that the start does not matter
hard_fits = {}
for tag, b0 in [("from b_2s", coef_soft), ("from b = 0", np.zeros_like(coef_soft))]:
    coef, res = kt.fit_two_step_operator_hard(stepper, active, alpha_bar, b0, lam_beta=LAM_BETA_2H, max_nfev=MAX_NFEV)
    hard_fits[tag] = np.asarray(coef)
    print(f"hard fit {tag:10s}: nfev {res.nfev:3d},  loss {0.5 * np.sum(res.fun ** 2) if False else res.cost:.3e} "
          f"(final),  converged {res.converged} (max cosine {res.max_cosine:.1e})")
coef_hard = hard_fits["from b_2s"]

slopes_soft, const_soft = active.linear_coefficients(coef_soft)
slopes_hard, const_hard = active.linear_coefficients(coef_hard)
true_terms = {"u*u_x": -A, "u_xxx": -B}
print("\nlinear atoms, in the original units of the features (2-step soft | 2-step hard | true):")
for name in slopes_soft:
    print(f"  {name:8s} {slopes_soft[name]: .5e}  {slopes_hard[name]: .5e}  {true_terms.get(name, 0.0): .5e}")
print(f"  {'constant':8s} {const_soft: .5e}  {const_hard: .5e}  {0.0: .5e}")
beta_hard = active.split(coef_hard)
for atom in active.atoms:
    if atom.size > 1:
        print(f"  {atom.name:16s} ({atom.size} coefficients; |beta| soft {np.linalg.norm(beta_soft[atom.name]):.2e}, "
              f"hard {np.linalg.norm(beta_hard[atom.name]):.2e})")

P_eval_hard = operator_at_true_features(selection.atoms, coef_hard)
print(f"\noperator nrmse at the true features: 2-step soft {nrmse(P_eval):.3e},  2-step hard {nrmse(P_eval_hard):.3e};   "
      f"|b_hard(from b_2s) - b_hard(from 0)| / |b_hard| = "
      f"{np.linalg.norm(hard_fits['from b_2s'] - hard_fits['from b = 0']) / np.linalg.norm(coef_hard):.1e}")

# ---- rollouts from the TRUE initial column over the training window, all in the same Crank-Nicolson stepper
alpha0_true = basis.interpolate(u_true[0])
U_floor = np.asarray(basis.values(true_stepper.rollout(alpha0_true, None, grid.n_steps)))   # true equation, same stepper
print(f"stepper floor: true equation rolled out from the true IC, rel. L2 error {kt.relative_l2(U_floor, u_true):.4f}")
U_roll = {m: np.asarray(basis.values(stepper.rollout(alpha0_true, jnp.asarray(b), grid.n_steps)))
          for m, b in [("ours soft", coef_soft), ("ours hard", coef_hard)]}

# =====================================================================================
# Visual check.
# Row 1: operator error P_hat(s_true) - u_t on the 100 x 100 grid (signed, one shared scale), title = nrmse.
# Row 2: truth at the nodes and the rollout of each learned operator from the true initial column (one shared scale),
#        title = relative L2 error of the rollout over all nodes.
# =====================================================================================
P_by = {"ours soft": P_eval, "ours hard": P_eval_hard}
elim = max(np.percentile(np.abs(P - ut_eval), 99.5) for P in P_by.values())
lim_t, lim_u = np.abs(ut_eval).max(), np.abs(u_true).max()

fig, ax = plt.subplots(2, 3, figsize=(18, 9.5), layout="constrained")
im = ph.field_map(ax[0, 0], g_t, g_x, ut_eval.reshape(nt_e, nx_e), lim_t, "true u_t")
fig.colorbar(im, ax=ax[0, 0], fraction=.046, pad=.02)
for j, (m, P) in enumerate(P_by.items(), start=1):
    im = ax[0, j].pcolormesh(ph.cell_edges(g_t), ph.cell_edges(g_x), (P - ut_eval).reshape(nt_e, nx_e).T, cmap="PuOr_r",
                             norm=ph.TwoSlopeNorm(0, -elim, elim), shading="flat", rasterized=True)
    ax[0, j].set_title(f"P_hat - u_t: {m}\nnrmse {nrmse(P):.3e}")
fig.colorbar(im, ax=ax[0, 2], fraction=.046, pad=.02, extend="both", label="P_hat - u_t (same scale)")

im = ph.field_map(ax[1, 0], grid.t, grid.x, u_true, lim_u, "true u at the nodes")
for j, (m, U) in enumerate(U_roll.items(), start=1):
    err_r = kt.relative_l2(U, u_true)
    im = ph.field_map(ax[1, j], grid.t, grid.x, np.clip(U, -3 * lim_u, 3 * lim_u), lim_u,
                      f"rollout from true IC: {m}\n" + (f"rel. L2 error {err_r:.3e}" if np.isfinite(err_r) and err_r < 10
                                                        else "diverged (Newton blew up)"))
fig.colorbar(im, ax=ax[1, 2], fraction=.046, pad=.02, extend="both", label="u (same scale)")
for a in ax.ravel():
    a.set(xlabel="t", ylabel="x")
plt.show()

# %% [markdown]
# 1-step soft constraint

# %%
# =====================================================================================
# 1-step method, SOFT constraint: state and operator fitted JOINTLY
# =====================================================================================
# Unknowns: alpha_0, ..., alpha_50 (all 51 columns) and b_A = (beta_m, m in A): the KMD operator with the atoms
#   selected by 2-step soft (fixed here),  P_hat(s) = Psi(s) b,  s = (u, u_x, u_xx, u_xxx).
# The library's standardisation and inducing points stay frozen at the Step 1 state's values.
# Loss:
#   J(alpha, b) = lambda_d     sum_{(n,k) in Omega} ( u_obs(t_n, x_k) - u_hat_n(x_k) )^2
#               + lambda_dyn   sum_{n=0}^{49} || e(alpha_{n+1}; b) ||^2
#               + lambda_alpha sum_{n=0}^{50} alpha_n^T K alpha_n  +  lambda_beta sum_{m in A} beta_m^T K_m beta_m
#               + lambda_addot / dt^4 sum_{n=1}^{49} || alpha_{n+1} - 2 alpha_n + alpha_{n-1} ||_K^2,
#   e(alpha_{n+1}; b) = K (alpha_{n+1} - alpha_n) - dt [ theta r(alpha_{n+1}; b) + (1 - theta) r(alpha_n; b) ],
#   r(alpha_n; b)_k = P_hat(s(x_k; alpha_n)) at every node (periodic: no boundary rows),  theta = THETA = 1/2.
# Whitened coordinates as in Step 1: w_n = L^T alpha_n (K + 1e-8 I = L L^T), so alpha_n^T K alpha_n ~ ||w_n||^2.
#   Unknowns p = (w_0, ..., w_50, b_A): 51 x 128 + n_coef(A).
# Optimiser: L-BFGS-B on the scalar loss with its exact gradient (jax.value_and_grad), started from the 2-step soft
#   solution (alpha_bar, coef_soft); the loss is divided by its starting value so the tolerances are scale-free
#   -> kt.fit_one_step_soft.
# Weights (hand-picked, no CV): lambda_d = LAM_D = 1 (reference), lambda_alpha = LAM_ALPHA = LAM_U (as in Step 1),
#   lambda_beta = LAM_BETA_1S tiny, lambda_dyn = LAM_DYN_1S, lambda_addot = LAM_ADDOT_1S (set below).
from dataclasses import replace

LAM_D, LAM_ALPHA, LAM_BETA_1S = 1.0, LAM_U, 1e-8                         # lambda_d, lambda_alpha, lambda_beta
LAM_DYN_GRID = np.r_[0, np.logspace(-4, 2, 13)]                          # lambda_dyn candidates (half decades)
LAM_ADDOT_GRID = np.unique(np.r_[0, 1e-10, 1e-9, 1e-8, cv_state.best, 1e-7, 1e-6])   # plus the Step 1 value
CV_MAXITER, MAXITER = 5000, 20000
weights_start = kt.OneStepWeights(data=LAM_D, dyn=1.0, alpha=LAM_ALPHA, beta=LAM_BETA_1S, addot=0.0)


def fit_1soft(train, weights, maxiter=MAXITER):
    """One-step soft fit on the observations `train`, started from the 2-step soft solution."""
    return kt.fit_one_step_soft(stepper, active, grid, train, alpha_bar, coef_soft, weights, maxiter)


# ---- hand-picked weights (no CV): lambda_dyn = 1 (data and e both in units of u), lambda_addot = 0 (the dynamics
#      replace the time prior, which smooths each fixed x over time and smears the moving solitons)
LAM_DYN_1S, LAM_ADDOT_1S = 1.0, 0.0
weights_1s = replace(weights_start, dyn=LAM_DYN_1S, addot=LAM_ADDOT_1S)
alpha_1s, coef_1s, res_1s = fit_1soft(obs, weights_1s)                   # one fit on all observations
alpha_1s, coef_1s = np.asarray(alpha_1s), np.asarray(coef_1s)

jets_1s = np.asarray(basis.jets(alpha_1s))                               # (51, 128, 4)
P_eval_1s = operator_at_true_features(selection.atoms, coef_1s)          # P_hat at the true features, 100 x 100 grid

slopes_1s, const_1s = active.linear_coefficients(coef_1s)
beta_1s = active.split(coef_1s)
print("linear atoms, in the original units of the features (2-step soft | 2-step hard | 1-step soft | true):")
for name in slopes_soft:
    print(f"  {name:8s} {slopes_soft[name]: .5e}  {slopes_hard[name]: .5e}  {slopes_1s[name]: .5e}  "
          f"{true_terms.get(name, 0.0): .5e}")
print(f"  {'constant':8s} {const_soft: .5e}  {const_hard: .5e}  {const_1s: .5e}  {0.0: .5e}")
for atom in active.atoms:
    if atom.size > 1:
        print(f"  {atom.name:16s} ({atom.size} coefficients; |beta| 2-step soft {np.linalg.norm(beta_soft[atom.name]):.2e}, "
              f"2-step hard {np.linalg.norm(beta_hard[atom.name]):.2e}, 1-step soft {np.linalg.norm(beta_1s[atom.name]):.2e})")
print(f"\nhand-picked lambda_dyn {LAM_DYN_1S:.1e}, lambda_addot {LAM_ADDOT_1S:.1e};  "
      f"L-BFGS nit {res_1s.nit}, {res_1s.message}")
names = ["u", "u_x", "u_xx", "u_xxx"]
for label, J in [("2-step (Step 1)", jets_bar), ("1-step soft", jets_1s)]:
    print(f"state rel. L2 error, {label:15s}: "
          + ",  ".join(f"{nm} {kt.relative_l2(J[..., q], jets_true[..., q]):.3f}" for q, nm in enumerate(names)))
print(f"operator nrmse at the true features: 2-step soft {nrmse(P_eval):.3e},  2-step hard {nrmse(P_eval_hard):.3e},  "
      f"1-step soft {nrmse(P_eval_1s):.3e}")

# =====================================================================================
# Visual check.
# Row 1: true u_t | P_hat(s_true) - u_t for 2-step soft, 2-step hard, 1-step soft (signed, shared scale, title = nrmse).
# Row 2: true u_xxx | u_xxx - true u_xxx at the nodes for the 2-step state (Step 1; 2-step hard keeps it) and the
#        1-step soft state (signed, shared scale) | u_xxx at t = 5 on a fine grid.
# =====================================================================================
fig = plt.figure(figsize=(23, 10), layout="constrained")
gs = fig.add_gridspec(2, 4)

# row 1: operator error at the true features
P_by = {"2-step soft": P_eval, "2-step hard": P_eval_hard, "1-step soft": P_eval_1s}
elim = max(np.percentile(np.abs(P - ut_eval), 99.5) for P in P_by.values())
a = fig.add_subplot(gs[0, 0])
im = ph.field_map(a, g_t, g_x, ut_eval.reshape(nt_e, nx_e), np.abs(ut_eval).max(), "true u_t")
fig.colorbar(im, ax=a, fraction=.046, pad=.02)
for j, (m, P) in enumerate(P_by.items(), start=1):
    a = fig.add_subplot(gs[0, j])
    im = a.pcolormesh(ph.cell_edges(g_t), ph.cell_edges(g_x), (P - ut_eval).reshape(nt_e, nx_e).T, cmap="PuOr_r",
                      norm=ph.TwoSlopeNorm(0, -elim, elim), shading="flat", rasterized=True)
    a.set(title=f"P_hat - u_t: {m}\nnrmse {nrmse(P):.3e}", xlabel="t", ylabel="x")
fig.colorbar(im, ax=a, fraction=.046, pad=.02, extend="both", label="P_hat - u_t (same scale)")

# row 2: u_xxx of the state
S_by = {"2-step (Step 1 state)": jets_bar[..., 3], "1-step soft": jets_1s[..., 3]}
true_xxx = jets_true[..., 3]
slim = max(np.percentile(np.abs(S - true_xxx), 99.5) for S in S_by.values())
a = fig.add_subplot(gs[1, 0])
im = ph.field_map(a, grid.t, grid.x, true_xxx, np.abs(true_xxx).max(), "true u_xxx at the nodes")
fig.colorbar(im, ax=a, fraction=.046, pad=.02)
for j, (m, S) in enumerate(S_by.items(), start=1):
    a = fig.add_subplot(gs[1, j])
    im = a.pcolormesh(ph.cell_edges(grid.t), ph.cell_edges(grid.x), (S - true_xxx).T, cmap="PuOr_r",
                      norm=ph.TwoSlopeNorm(0, -slim, slim), shading="flat", rasterized=True)
    a.set(title=f"u_xxx - true u_xxx: {m}\nrel. L2 error {kt.relative_l2(S, true_xxx):.3f}", xlabel="t", ylabel="x")
fig.colorbar(im, ax=a, fraction=.046, pad=.02, extend="both", label="u_xxx error (same scale)")

a = fig.add_subplot(gs[1, 3])                                            # snapshot at t = 5 on a fine grid
n_snap = 25
B3 = basis.matrix(x_fine, 3)
a.plot(x_fine, truth.derivative(grid.t[n_snap], x_fine, 3)[0], c=ph.NEUTRAL, lw=5, label="truth")
a.plot(x_fine, B3 @ alpha_bar[n_snap], c=ph.BLUE, lw=1.5, label="2-step (Step 1)")
a.plot(x_fine, B3 @ alpha_1s[n_snap], c=ph.ORANGE, lw=1.5, label="1-step soft")
a.set(title=f"u_xxx at t = {grid.t[n_snap]:.0f}", xlabel="x", ylabel="u_xxx")
a.legend(fontsize=8)
plt.show()

# %% [markdown]
# 1-step hard constraint

# %%
# =====================================================================================
# 1-step method, HARD constraint: every column is the Crank-Nicolson rollout of (alpha_0, b)
# =====================================================================================
# Unknowns: alpha_0 (128) and b_A = (beta_m, m in A): the KMD operator with the atoms selected by 2-step soft (fixed
#   here),  P_hat(s) = Psi(s) b.  Every later column is alpha_n = Phi^n(alpha_0; b), with the step map Phi of the
#   2-step hard cell (`stepper`: 6 Newton iterations on e(alpha_{n+1}; b) = 0), so e = 0 holds at every step.
# Loss: only the free variables alpha_0 and b are penalised (the rolled-out columns are not; the time prior has no
#   form on alpha_0 alone and is dropped, so there is no lambda_addot and no CV here):
#   J(alpha_0, b) = lambda_d     sum_{(n,k) in Omega} ( u_obs(t_n, x_k) - u_hat_n(x_k) )^2
#                 + lambda_alpha alpha_0^T K alpha_0  +  lambda_beta sum_{m in A} beta_m^T K_m beta_m,
#   alpha_n = Phi^n(alpha_0; b),  u_hat_n(x_k) = (K alpha_n)_k.
# Least-squares form, residual vector [ sqrt(lambda_d) (u_hat_n(x_k) - u_obs) ;  sqrt(lambda_alpha) w_0 ;
#   sqrt(lambda_beta) L_A^T b_A ],  w_0 = L^T alpha_0 (K + 1e-8 I = L L^T),  blockdiag(K_m) = L_A L_A^T;
#   trust-region Gauss-Newton (scipy least_squares) with the exact Jacobian (jax.jacfwd through the rollout)
#   -> kt.fit_one_step_hard.  Converged = max cosine between residual and Jacobian columns <= 1e-3.
# Start: the 1-step soft solution (alpha_1s[0], coef_1s); (alpha_bar_0, b_2h) is run as a check that the start does
#   not matter.
# Weights: lambda_d = LAM_D = 1, lambda_alpha = LAM_ALPHA (= LAM_U), lambda_beta = LAM_BETA_1H = 1e-8 (fixed).
LAM_BETA_1H, MAX_NFEV_1H = 1e-8, 100
weights_1h = kt.OneStepWeights(data=LAM_D, alpha=LAM_ALPHA, beta=LAM_BETA_1H)

fits_1h = {}
for tag, a0, b0 in [("from 1-step soft", alpha_1s[0], coef_1s), ("from (alpha_bar_0, b_2h)", alpha_bar[0], coef_hard)]:
    alpha, coef, res = kt.fit_one_step_hard(stepper, active, grid, obs, a0, b0, weights_1h, max_nfev=MAX_NFEV_1H)
    fits_1h[tag] = (np.asarray(alpha), np.asarray(coef), res)
    print(f"1-step hard {tag:26s}: nfev {res.nfev:3d},  loss {res.cost:.3e} (final),  "
          f"converged {res.converged} (max cosine {res.max_cosine:.1e})")
alpha_1h, coef_1h, res_1h = fits_1h["from 1-step soft"]
jets_1h = np.asarray(basis.jets(alpha_1h))                               # (51, 128, 4)
P_eval_1h = operator_at_true_features(selection.atoms, coef_1h)

slopes_1h, const_1h = active.linear_coefficients(coef_1h)
beta_1h = active.split(coef_1h)
print("\nlinear atoms, in the original units of the features (2-step soft | 2-step hard | 1-step soft | 1-step hard | true):")
for name in slopes_soft:
    print(f"  {name:8s} {slopes_soft[name]: .5e}  {slopes_hard[name]: .5e}  {slopes_1s[name]: .5e}  "
          f"{slopes_1h[name]: .5e}  {true_terms.get(name, 0.0): .5e}")
print(f"  {'constant':8s} {const_soft: .5e}  {const_hard: .5e}  {const_1s: .5e}  {const_1h: .5e}  {0.0: .5e}")
for atom in active.atoms:
    if atom.size > 1:
        print(f"  {atom.name:16s} ({atom.size} coefficients; |beta| 2-step soft {np.linalg.norm(beta_soft[atom.name]):.2e}, "
              f"2-step hard {np.linalg.norm(beta_hard[atom.name]):.2e}, 1-step soft {np.linalg.norm(beta_1s[atom.name]):.2e}, "
              f"1-step hard {np.linalg.norm(beta_1h[atom.name]):.2e})")
coef_check = fits_1h["from (alpha_bar_0, b_2h)"][1]
print(f"\n|b(from 1-step soft) - b(from (alpha_bar_0, b_2h))| / |b| = "
      f"{np.linalg.norm(coef_1h - coef_check) / np.linalg.norm(coef_1h):.1e}")
print(f"state rel. L2 error, {'1-step hard':15s}: "
      + ",  ".join(f"{nm} {kt.relative_l2(jets_1h[..., q], jets_true[..., q]):.3f}" for q, nm in enumerate(names)))
print(f"operator nrmse at the true features: 2-step soft {nrmse(P_eval):.3e},  2-step hard {nrmse(P_eval_hard):.3e},  "
      f"1-step soft {nrmse(P_eval_1s):.3e},  1-step hard {nrmse(P_eval_1h):.3e}")

# ---- rollouts from the TRUE initial column over the training window (same stepper), for 1-step soft and 1-step hard
for m, b in [("ours 1-step soft", coef_1s), ("ours 1-step hard", coef_1h)]:
    U_roll[m] = np.asarray(basis.values(stepper.rollout(alpha0_true, jnp.asarray(b), grid.n_steps)))
print("rollout from the true IC, rel. L2 error over the training window:  "
      + ",  ".join(f"{m} {kt.relative_l2(U, u_true):.2e}" for m, U in U_roll.items())
      + f"  (true equation, same stepper: {kt.relative_l2(U_floor, u_true):.2e})")

# =====================================================================================
# Visual check.
# Row 1: true u_t | P_hat(s_true) - u_t for 2-step hard, 1-step soft, 1-step hard (signed, shared scale, title = nrmse).
# Row 2: learned component of u*u_x (sum over the selected atoms on u*u_x of phi_m(z) beta_m) against the true
#        -A u*u_x, all centred (the constant atom absorbs the means) | u_xxx - true u_xxx at the nodes for the same
#        three (2-step hard keeps the Step 1 state; signed, shared scale).
# Row 3: true u at the nodes | 1-step hard state (= its own rollout from the fitted alpha_0) - truth
#        | rollout from the TRUE initial column minus truth: 1-step soft, 1-step hard (signed, shared scale).
# =====================================================================================
P_by = {"2-step hard": P_eval_hard, "1-step soft": P_eval_1s, "1-step hard": P_eval_1h}
S_by = {"2-step hard (Step 1 state)": jets_bar[..., 3], "1-step soft": jets_1s[..., 3], "1-step hard": jets_1h[..., 3]}
coef_by = {"2-step soft": (coef_soft, ph.NEUTRAL), "1-step soft": (coef_1s, ph.BLUE), "1-step hard": (coef_1h, ph.ORANGE)}
uerr = {"1-step hard state (own rollout)": alpha_1h @ basis.K - u_true,
        "rollout from true IC: 1-step soft": U_roll["ours 1-step soft"] - u_true,
        "rollout from true IC: 1-step hard": U_roll["ours 1-step hard"] - u_true}
elim = max(np.percentile(np.abs(P - ut_eval), 99.5) for P in P_by.values())
slim = max(np.percentile(np.abs(S - true_xxx), 99.5) for S in S_by.values())
ulim = max(np.percentile(np.abs(E), 99.5) for E in uerr.values())

fig, ax = plt.subplots(3, 4, figsize=(23, 15), layout="constrained")
im = ph.field_map(ax[0, 0], g_t, g_x, ut_eval.reshape(nt_e, nx_e), np.abs(ut_eval).max(), "true u_t")
fig.colorbar(im, ax=ax[0, 0], fraction=.046, pad=.02)
for j, (m, P) in enumerate(P_by.items(), start=1):
    im = ax[0, j].pcolormesh(ph.cell_edges(g_t), ph.cell_edges(g_x), (P - ut_eval).reshape(nt_e, nx_e).T, cmap="PuOr_r",
                             norm=ph.TwoSlopeNorm(0, -elim, elim), shading="flat", rasterized=True)
    ax[0, j].set(title=f"P_hat - u_t: {m}\nnrmse {nrmse(P):.3e}", xlabel="t", ylabel="x")
fig.colorbar(im, ax=ax[0, 3], fraction=.046, pad=.02, extend="both", label="P_hat - u_t (same scale)")

a = ax[1, 0]                                                             # component of u*u_x
i_uux = [f.name for f in library.features].index("u*u_x")
G_true = np.asarray(library.raw_features(jnp.asarray(jets_true))).reshape(-1, len(library.features))
gg = np.linspace(G_true[:, i_uux].min(), G_true[:, i_uux].max(), 200)
z = (gg - library.center[i_uux]) / library.scale[i_uux]
tr = -A * gg
a.plot(gg, tr - tr.mean(), c=ph.MUTED, lw=4, alpha=.6, label="true: -A u*u_x")
for m, (coef, color) in coef_by.items():
    beta = active.split(coef)
    comp = sum((np.asarray(atom.phi(jnp.asarray(z))) @ beta[atom.name] for atom in active.atoms if atom.feature == i_uux),
               np.zeros_like(gg))
    a.plot(gg, comp - comp.mean(), c=color, lw=1.8, label=m)
a.set(title="learned component of u*u_x (centred)\n" + ", ".join(at.name for at in active.atoms if at.feature == i_uux),
      xlabel="u*u_x (range of the true features)", ylabel="contribution to P_hat")
a.legend(fontsize=8)
for j, (m, S) in enumerate(S_by.items(), start=1):
    im = ax[1, j].pcolormesh(ph.cell_edges(grid.t), ph.cell_edges(grid.x), (S - true_xxx).T, cmap="PuOr_r",
                             norm=ph.TwoSlopeNorm(0, -slim, slim), shading="flat", rasterized=True)
    ax[1, j].set(title=f"u_xxx - true u_xxx: {m}\nrel. L2 error {kt.relative_l2(S, true_xxx):.3f}", xlabel="t", ylabel="x")
fig.colorbar(im, ax=ax[1, 3], fraction=.046, pad=.02, extend="both", label="u_xxx error (same scale)")

im = ph.field_map(ax[2, 0], grid.t, grid.x, u_true, np.abs(u_true).max(), "true u at the nodes")
fig.colorbar(im, ax=ax[2, 0], fraction=.046, pad=.02)
for j, (m, E) in enumerate(uerr.items(), start=1):
    im = ax[2, j].pcolormesh(ph.cell_edges(grid.t), ph.cell_edges(grid.x), E.T, cmap="PuOr_r",
                             norm=ph.TwoSlopeNorm(0, -ulim, ulim), shading="flat", rasterized=True)
    ax[2, j].set(title=f"{m} - truth\nrel. L2 error {kt.relative_l2(E + u_true, u_true):.3e}", xlabel="t", ylabel="x")
fig.colorbar(im, ax=ax[2, 3], fraction=.046, pad=.02, extend="both", label="u error (same scale)")
plt.show()

# %%
# =====================================================================================
# Comparison: all our methods (training window t in [0, T_TRAIN])
# =====================================================================================
# Methods (KMD operator P_hat(s) = Psi(s) b on the atoms selected by 2-step soft, s = (u, u_x, u_xx, u_xxx)):
#   2-step soft   state = alpha_bar (Step 1),               operator = coef_soft
#   2-step hard   state = alpha_bar (same Step 1 state),    operator = coef_hard
#   1-step soft   state = alpha_1s,                         operator = coef_1s
#   1-step hard   state = alpha_1h = Phi^n(alpha_0; b_1h),  operator = coef_1h
# Fields on the 51 time columns t_n x 400 periodic points in x (ours only exists at the columns).
#   R_eql      : ||P_hat(s_true) - u_t|| / ||u_t||, P_hat at the TRUE features on the 100 x 100 grid (earlier cells)
#   R_filter   : ||u_hat - u|| / ||u|| on a 300 x 300 grid over [0, T_TRAIN] x [X_MIN, X_MIN + PERIOD); our columns are
#                interpolated in t by a cubic spline (adds interpolation error)
#   u at t_n   : ||u_hat - u|| / ||u|| at the 51 columns only (no interpolation)
#   u_xxx      : the method's state u_hat_xxx vs true u_xxx
#   P_hat      : the model's u_t along its OWN state, P_hat(u_hat, ..., u_hat_xxx), vs true u_t (state and operator error)
#   rollout    : the learned operator marched from the TRUE initial column with our CN stepper (no data after t = 0)
# The reference is a Fourier series (exact x-derivatives everywhere, u_t from the PDE), so every metric uses the whole
#   window: no evaluation box (unlike Burgers, whose spline reference was unreliable at the edges).
# Colours: one hue per family (2-step blue, 1-step aqua); soft = solid line / circle, hard = dashed line / square.
#   Error maps show |prediction - truth| on a log scale, one shared scale per figure.
from matplotlib.colors import LogNorm
from scipy.interpolate import CubicSpline

xf = X_MIN + PERIOD * np.arange(400) / 400                              # 400 periodic points in x
Bf = [np.asarray(basis.matrix(xf, q)) for q in range(4)]               # (400, 128): basis and x-derivatives on xf
METHODS = {"2-step soft": (alpha_bar, coef_soft), "2-step hard": (alpha_bar, coef_hard),
           "1-step soft": (alpha_1s, coef_1s), "1-step hard": (alpha_1h, coef_1h)}
SHARED = "2-step (shared state)"                                       # 2-step soft and hard share the Step 1 state
STYLE = {"2-step soft": (ph.BLUE, "-", "o"), "2-step hard": (ph.BLUE, "--", "s"),
         "1-step soft": (ph.AQUA, "-", "o"), "1-step hard": (ph.AQUA, "--", "s"), SHARED: (ph.BLUE, "-", "o")}
P_AT_TRUE = {"2-step soft": P_eval, "2-step hard": P_eval_hard, "1-step soft": P_eval_1s, "1-step hard": P_eval_1h}

# ---- truth on (t_n, xf), all (51, 400)
jets_f = truth.jets(grid.t, xf, max_order=3)                           # (51, 400, 4)
tru_u, tru_uxxx, tru_ut = jets_f[..., 0], jets_f[..., 3], truth.u_t(grid.t, xf)


def roll_fine(stepper_, coef):
    """Rollout from the true initial column on (t_n, xf); columns after a blow-up are NaN. Returns (field, first
    diverged column or None)."""
    F = np.asarray(stepper_.rollout(alpha0_true, None if coef is None else jnp.asarray(coef), grid.n_steps)) @ Bf[0].T
    bad = ~np.all(np.isfinite(F), 1) | (np.abs(F).max(1) > 10 * np.abs(tru_u).max())
    n_div = int(np.argmax(bad)) if bad.any() else None
    if n_div is not None:
        F[n_div:] = np.nan
    return F, n_div


U_rec, Uxxx_rec, Ut_rec, U_rol, div_at = {}, {}, {}, {}, {}
for m, (al, coef) in METHODS.items():
    S = np.stack([al @ B.T for B in Bf], -1)                           # (51, 400, 4): (u, u_x, u_xx, u_xxx) of the state
    U_rec[m], Uxxx_rec[m] = S[..., 0], S[..., 3]
    Ut_rec[m] = np.asarray(active.evaluate(jnp.asarray(S), jnp.asarray(coef)))   # P_hat along the method's own state
    U_rol[m], div_at[m] = roll_fine(stepper, coef)
U_floor_f, _ = roll_fine(true_stepper, None)                           # true equation, same stepper: the floor

# ---- R_filter grid: 300 x 300 over the training window
t300, x300 = np.linspace(0, T_TRAIN, 300), X_MIN + PERIOD * np.arange(300) / 300
tru_300, B300 = truth.u(t300, x300), np.asarray(basis.matrix(x300))
R_filter = {m: np.sqrt(np.mean((CubicSpline(grid.t, al @ B300.T, axis=0)(t300) - tru_300) ** 2) / np.mean(tru_300 ** 2))
            for m, (al, _) in METHODS.items()}


def rel_all(F, T):                                                     # relative L2 error over the whole grid
    return np.sqrt(np.sum((F - T) ** 2) / np.sum(T ** 2)) if np.all(np.isfinite(F)) else np.nan


def rel_t(F, T):                                                       # relative L2 error per time column
    return np.sqrt(np.sum((F - T) ** 2, 1) / np.sum(T ** 2, 1))


metrics = {m: {"R_eql": nrmse(P_AT_TRUE[m]), "R_filter": R_filter[m], "u": rel_all(U_rec[m], tru_u),
               "u_xxx": rel_all(Uxxx_rec[m], tru_uxxx), "P_hat": rel_all(Ut_rec[m], tru_ut),
               "rollout": rel_all(U_rol[m], tru_u)} for m in METHODS}
PANEL = {"R_eql": r"$R_{eql}$: $\hat P$ at true features vs $u_t$", "R_filter": r"$R_{filter}$: $\hat u$ vs $u$ (300 x 300)",
         "u": r"$\hat u$ vs $u$ at the 51 columns $t_n$", "u_xxx": r"$\hat u_{xxx}$ vs $u_{xxx}$",
         "P_hat": r"$\hat P$ along own state vs $u_t$", "rollout": r"rollout from true $u(0)$ vs $u$"}
floor_roll = rel_all(U_floor_f, tru_u)

# ---- printed table (same numbers as the summary figure)
cols_ = list(PANEL)
print("relative L2 error (training window)")
print(f"  {'':18s}" + "".join(f"{c:>10s}" for c in cols_))
for m, v in metrics.items():
    print(f"  {m:18s}" + "".join(f"{('diverged' if np.isnan(v[c]) else f'{v[c]:.2g}'):>10s}" for c in cols_))
print(f"  {'floor (true eq.)':18s}" + "".join(f"{(f'{floor_roll:.2g}' if c == 'rollout' else ''):>10s}" for c in cols_))

# =====================================================================================
# Figure 1: summary. One panel per metric, one row per method (log x; lower is better).
# =====================================================================================
names_m = list(METHODS)
fig, ax = plt.subplots(1, len(cols_), figsize=(3.9 * len(cols_), 3.6), layout="constrained", sharey=True)
for a, c in zip(ax, cols_):
    vals = np.array([metrics[m][c] for m in names_m])
    fin = vals[np.isfinite(vals)]
    lo, hi = fin.min() / 3, fin.max() * 8
    for y, m in enumerate(names_m):
        col, ls, mk = STYLE[m]
        if np.isfinite(vals[y]):
            a.plot([lo, vals[y]], [y, y], c=ph.GRIDLINE, lw=1, zorder=1)
            a.plot(vals[y], y, mk, ms=10, c=col, mec=ph.SURFACE, mew=2, zorder=3)
            a.text(vals[y] * 1.25, y, f"{vals[y]:.2g}", va="center", fontsize=9, color=ph.INK)
        else:
            a.text(lo * 1.1, y, "diverged", va="center", fontsize=9, color=ph.MUTED, style="italic")
    if c == "rollout":
        a.axvline(floor_roll, c=ph.MUTED, ls=":", lw=1.2)
        a.text(floor_roll * 1.1, -0.75, f"floor {floor_roll:.2g}", fontsize=8, color=ph.MUTED, va="center")
    a.set_xscale("log"); a.set_xlim(lo, hi)
    a.set_title(PANEL[c], fontsize=10.5, loc="left"); a.set_xlabel("relative L2 error (log)")
ax[0].set_yticks(range(len(names_m)), names_m); ax[0].set_ylim(len(names_m) - 0.5, -1.1)
fig.suptitle("Relative L2 error over the training window (lower is better)", x=0.01, ha="left", fontsize=12)
plt.show()

# =====================================================================================
# Figures 2-5: space-time maps. Row 1: truth and each method's field; row 2: error over time (all methods, log) and each
# method's |error| map on a log colour scale (shared by all methods in the figure). Columns = methods (2-step soft and
# hard share one state, so the state figures show it once). Titles give the relative L2 error.
# =====================================================================================
te_f, xe_f = ph.cell_edges(grid.t), ph.cell_edges(xf)


def spacetime(title, truth_f, fields, err, floor=None, show_obs=False):   # err: label of the error, e.g. r"$|\hat u - u|$"
    ms = list(fields)
    lim = np.abs(truth_f).max()
    errs = {m: fields[m] - truth_f for m in ms}
    emax = np.nanmax([np.nanpercentile(np.abs(e), 99.5) for e in errs.values() if np.isfinite(e).any()])
    enorm = LogNorm(emax * 1e-3, emax)                                 # 3 decades of |error|
    fig, ax = plt.subplots(2, len(ms) + 1, figsize=(4.6 * (len(ms) + 1), 8.6), layout="constrained")
    im = ph.field_map(ax[0, 0], grid.t, xf, np.clip(truth_f, -3 * lim, 3 * lim), lim, "truth" + (" + obs." if show_obs else ""))
    if show_obs:
        ax[0, 0].scatter(grid.t[obs.n], grid.x[obs.k], s=3, c=ph.INK, lw=0)
    for j, m in enumerate(ms, start=1):
        F = fields[m]
        ph.field_map(ax[0, j], grid.t, xf, np.clip(np.nan_to_num(F), -3 * lim, 3 * lim), lim, m)
        im2 = ax[1, j].pcolormesh(te_f, xe_f, np.clip(np.abs(errs[m]), emax * 1e-3, None).T, cmap="Purples", norm=enorm,
                                  shading="flat", rasterized=True)
        e = rel_all(F, truth_f)
        ax[1, j].set_title(f"{err}\ndiverged at t = {grid.t[div_at[m]]:.1f}" if np.isnan(e) and div_at.get(m) is not None
                           else f"{err}\nrel. L2: {e:.2g}")
    fig.colorbar(im, ax=ax[0, -1], fraction=.046, pad=.02, label=title.split(":")[0])
    fig.colorbar(im2, ax=ax[1, -1], fraction=.046, pad=.02, extend="both", label=err + "  (log scale)")
    a = ax[1, 0]                                                       # error over time, all methods
    for m in ms:
        c, ls, mk = STYLE[m]
        a.semilogy(grid.t, rel_t(fields[m], truth_f), ls=ls, marker=mk, ms=3, c=c, lw=2, label=m)
    if floor is not None:
        a.semilogy(grid.t, rel_t(floor, truth_f), c=ph.MUTED, lw=3, alpha=.5, label="floor (true eq.)")
    a.set(title="rel. L2 error at each $t_n$", xlabel="t", ylabel="rel. L2 error (log)"); a.legend(fontsize=7)
    for a_ in ax.ravel():
        if a_ is not ax[1, 0]:
            a_.set(xlabel="t", ylabel="x")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    plt.show()


state_of = lambda R: {SHARED: R["2-step soft"], "1-step soft": R["1-step soft"], "1-step hard": R["1-step hard"]}
spacetime(r"State $\hat u$: each method's reconstructed $u$ (row 1) and its error (row 2)", tru_u, state_of(U_rec),
          r"$|\hat u - u|$", show_obs=True)
spacetime(r"State $\hat u_{xxx}$: each method's reconstructed $u_{xxx}$ and its error", tru_uxxx, state_of(Uxxx_rec),
          r"$|\hat u_{xxx} - u_{xxx}|$")
spacetime(r"$\hat P(\hat u, \dots, \hat u_{xxx})$ along each method's own state vs $u_t$", tru_ut, Ut_rec,
          r"$|\hat P - u_t|$")
spacetime(r"Rollout: each learned $\hat P$ stepped from the true $u(0)$ with our CN stepper (no data after t = 0)",
          tru_u, U_rol, r"$|\hat u_{roll} - u|$", floor=U_floor_f)
