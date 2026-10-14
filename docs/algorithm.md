# Kernel time-stepping equation learning with a KMD operator library

This document describes, as an algorithm, the method labelled "ours" in
[`archive/keql_fig3B_original.ipynb`](../archive/keql_fig3B_original.ipynb), and how the
[`kernel_timestepping`](../kernel_timestepping/) package implements it. The parts of that notebook that run
Kernel Equation Learning (`keql_tools`) for comparison are not covered.

It was written from the notebook's code. The notebook's comments cite pages of a separate note ("note p.3" etc.) for the
residual and the losses; that note was not available here.

**In one paragraph.** The task is to learn the right-hand side of an evolution equation $u_t = P(u, u_x, u_{xx}, \dots)$
from sparse observations of one solution. The state is a spatial kernel expansion in every time column. The operator
$\hat P$ is a sum of *atoms*, each a function of one feature of the state taken from the RKHS of one scalar kernel
(the "KMD" library: kernel mode decomposition; the notebook does not spell the acronym out). State and operator are tied
together by a Crank-Nicolson step. The atoms are chosen by backward elimination with cross-validation, and the learned
operator is used by rolling the same Crank-Nicolson step forward.

Contents: [1 Setting](#1-setting) · [2 State](#2-state-representation) · [3 Operator library](#3-operator-library-kmd) ·
[4 Time step](#4-the-time-step-that-couples-them) · [5 Two-step method](#5-two-step-method) ·
[6 One-step methods](#6-one-step-methods) · [7 Summary and defaults](#7-summary-and-defaults) ·
[8 The package](#8-the-package) · [9 Checks and observations](#9-checks-and-observations)

---

## 1. Setting

* Unknown equation $u_t = P(u, u_x, u_{xx}, \dots)$ on $x \in (0, 1)$, $t \in [0, 1]$, with boundary values that do not
  change in time. In the notebook the truth is $P = \kappa u_{xx} + \alpha\, u u_x$ with $u = 0$ on the boundary.
* Grid: $N_T = 26$ time columns $t_n = n\,\Delta t$, $\Delta t = 0.04$, and $N_X = 31$ spatial nodes $x_k$ shared by
  all columns (an equal blend of Chebyshev and uniform nodes). $x_0, x_{N_X-1}$ are the boundary; the equation is
  collocated at the $N_X - 2$ interior nodes.
* Data: values $y_{nk} = u(t_n, x_k)$ at a set $\Omega$ of nodes: the whole initial column, the two boundary nodes of
  every column, and a random sample of interior nodes (60 in experiment 3B, 100 in 3A).

## 2. State representation

Every time column is a kernel expansion centred at the nodes, with a spatial kernel $k$ (Gaussian RBF, lengthscale
$\ell_x$):

$$u_n(x) = \sum_{l} k(x, x_l)\, \alpha_{n,l}, \qquad \alpha_n \in \mathbb{R}^{N_X} .$$

Everything needed from the state is linear in $\alpha_n$:

| Quantity | Formula | Matrix |
|---|---|---|
| nodal values | $u_n(x_k) = (K\alpha_n)_k$ | $K_{kl} = k(x_k, x_l)$, $N_X \times N_X$ |
| $q$-th $x$-derivative at interior nodes | $\partial_x^q u_n(x_k) = (D_q\alpha_n)_k$ | $(D_q)_{kl} = \partial_x^q k(x, x_l)\vert_{x = x_k}$, $(N_X-2) \times N_X$ |

The **jet** of column $n$ at interior node $k$ is $s(x_k; \alpha_n) = (u, u_x, u_{xx}, u_{xxx})$, i.e. $D_0 \dots D_3$
applied to $\alpha_n$.

*Whitening.* With $K + 10^{-8} I = L L^\top$ and $\alpha_n = L^{-\top} w_n$, the RKHS norm $\alpha_n^\top K \alpha_n$
equals $\lVert w_n \rVert^2$ up to the nugget. All state regularisers are written in $w$.

*Lengthscale.* The notebook takes $\ell_x$ from Kernel Equation Learning's hyperparameter fit so that both methods
use the same kernel: a space-time RBF $\exp(-\tfrac12 (t-t')^2/\ell_t^2 - \tfrac12 (x-x')^2/\ell_x^2)$ of unit variance
is fitted to the observations by maximum marginal likelihood, and the state kernel uses $0.75\,\ell_x$ ($\ell_t$ is
unused: the state kernel acts in space only).

## 3. Operator library (KMD)

**Features.** Six scalar functions of the jet, $g(s) = (u,\ u_x,\ u_{xx},\ u_{xxx},\ u u_x,\ u^2)$, each standardised
with the mean $c_i$ and standard deviation $\sigma_i$ it has over all interior points of the Step-1 state (section 5):
$z_i = (g_i(s) - c_i)/\sigma_i$.

**Atoms.** An atom $m$ is a pair (feature $i(m)$, scalar kernel $k_m$), represented with inducing points
$\zeta_{m1}, \dots, \zeta_{mJ}$:

$$\hat P_m(s) = \sum_{j=1}^{J} k_m\big(z_{i(m)}, \zeta_{mj}\big)\, \beta_{mj} = \phi_m\big(z_{i(m)}\big)^\top \beta_m,
\qquad \lVert \hat P_m \rVert_{k_m}^2 = \beta_m^\top K_m \beta_m, \quad K_m = \big[k_m(\zeta_{mj}, \zeta_{mj'})\big] .$$

| Kernel on the standardised feature | $k(z, z')$ | Coefficients | Meaning |
|---|---|---|---|
| linear | $z z'$ | 1 ($\phi(z) = z$, $K_m = 1$) | a term linear in the feature |
| Gaussian RBF | $\exp(-(z - z')^2/2)$ | $J = 15$ | a smooth function of the feature |
| Matérn 3/2 | $(1 + \sqrt3\,\lvert z - z'\rvert)\, e^{-\sqrt3 \lvert z - z'\rvert}$ | $J = 15$ | a rougher function of the feature |
| constant | $1$ | 1 | the constant of $\hat P$; never pruned |

The inducing points of an RBF or Matérn atom are 15 evenly spaced values between the smallest and largest value of
its standardised feature on the Step-1 state. With 6 features and 3 kernels the library has $1 + 18 = 19$ atoms and
$1 + 6 + 2 \cdot 6 \cdot 15 = 187$ coefficients.

**Operator.** With $b = (\beta_1, \dots, \beta_M)$ stacked and $\Psi(s)$ the row of all $\phi_m$ side by side,

$$\hat P(s) = \sum_m \hat P_m(s) = \Psi(s)\, b, \qquad \sum_m \lVert \hat P_m \rVert^2 = b^\top K_{\rm reg}\, b,
\quad K_{\rm reg} = \mathrm{blockdiag}(K_m) .$$

The true operator of the notebook is in the library exactly: the constant atom plus the linear atoms on $u_{xx}$ and
$u u_x$. The standardisation and the inducing points are fixed once and never refitted.

## 4. The time step that couples them

Right-hand side on all nodes, zero in the two boundary rows:

$$r(\alpha; b) = \big[\,0;\ \hat P\big(s(x_k; \alpha)\big)_{k = 1..N_X-2};\ 0\,\big] .$$

**Dynamic residual** of step $n \to n+1$, with $\theta = \tfrac12$ (Crank-Nicolson):

$$e(\alpha_{n+1}, \alpha_n; b) = K(\alpha_{n+1} - \alpha_n) - \Delta t\,\big[\theta\, r(\alpha_{n+1}; b) + (1 - \theta)\, r(\alpha_n; b)\big] .$$

Its boundary rows contain no operator term; they say that the boundary values do not move.

**Step map** $\alpha_{n+1} = \Phi(\alpha_n; b)$: solve $e = 0$ by 6 Newton iterations started at $\alpha_n$, with
Jacobian $J = K - \Delta t\,\theta\, \partial r/\partial \alpha$ (forward-mode autodiff through $\hat P$ and the $D_q$).
The iteration count is fixed, so $\Phi$ is an ordinary differentiable function of $\alpha_n$ and $b$; the
hard-constraint fits below differentiate through it.

**Rollout**: $\alpha_n = \Phi^n(\alpha_0; b)$. To roll out from a given initial condition, $\alpha_0 = K^{-1} u(0, x_\cdot)$.

## 5. Two-step method

### Step 1: fit the state

All columns together:

$$\min_{\alpha}\ \sum_{(n,k) \in \Omega} \big(y_{nk} - (K\alpha_n)_k\big)^2
 + \lambda_u \sum_{n} \alpha_n^\top K \alpha_n
 + \frac{\lambda_{\ddot a}}{\Delta t^4} \sum_{n=1}^{N_T-2} \lVert \alpha_{n+1} - 2\alpha_n + \alpha_{n-1} \rVert_K^2 .$$

The last term is a prior on the discrete acceleration $u_{tt}$ and is the only coupling between columns. It smooths
the state in time without shrinking the rate $u_t$, which is what Step 2 fits. $\lambda_{\ddot a} = 0$ fits each column
on its own.

*Solver.* In whitened coordinates this is one ridge regression for $w = (w_0, \dots, w_{N_T-1})$,

$$\min_w \lVert M_{\rm aug}\, w - y_{\rm aug} \rVert^2 + \lambda_u \lVert w \rVert^2, \qquad
M_{\rm aug} = \begin{bmatrix} M \\ \sqrt{\lambda_{\ddot a}}\,\Delta t^{-2}\,(D_{tt} \otimes I) \end{bmatrix}, \quad
y_{\rm aug} = \begin{bmatrix} y \\ 0 \end{bmatrix},$$

where row $i$ of $M$ holds $(K L^{-\top})_{k_i, \cdot}$ in the block of column $n_i$ and $D_{tt}$ is the second
difference in time. It is solved through the SVD of $M_{\rm aug}$.

*Choice of $\lambda_{\ddot a}$.* 5-fold cross-validation over $\lbrace 0\rbrace \cup \lbrace 10^{-12}, \dots, 10^{-1}\rbrace$ (23 log-spaced
values). Only interior observations are held out; boundary and initial-column observations stay in every fit. The
score is the RMS prediction error pooled over the held-out observations. The result is the state $\bar\alpha$.

### Step 2 (soft constraint): fit the operator on the frozen state

With $\alpha = \bar\alpha$ the residual is linear in $b$. At the interior nodes

$$e_n = d_n - A_n b, \qquad d_n = \big(K\bar\alpha_{n+1} - K\bar\alpha_n\big)_{\rm int}, \qquad
A_n = \Delta t\,\big[\theta\,\Psi(\bar s_{n+1}) + (1 - \theta)\,\Psi(\bar s_n)\big],$$

and for a set $\mathcal A$ of active atoms (the corresponding columns of $A$)

$$J(b) = \lambda_{\rm dyn} \sum_n \lVert e_n \rVert^2 + \lambda_\beta \sum_{m \in \mathcal A} \beta_m^\top K_m \beta_m
\ \Longrightarrow\
b = \big(\lambda_{\rm dyn} A^\top A + \lambda_\beta K_{\rm reg}\big)^{-1} \lambda_{\rm dyn} A^\top d ,$$

with $\lambda_{\rm dyn} = 1$ and a relative ridge $10^{-12}\,\mathrm{tr}(H)/\dim$ added to the matrix $H$ being
inverted.

*Choice of $\lambda_\beta$.* Cross-validation over 5 contiguous blocks of time steps, on the grid
$10^{-12}, \dots, 10^{2}$ (15 values). Score: RMS of $e_n$ on the held-out steps.

### Atom selection: backward elimination and the one-standard-error rule

1. Start with all atoms, $\mathcal A_0$.
2. For every non-constant atom $m \in \mathcal A_j$, compute the CV score of $\mathcal A_j \setminus \lbrace m\rbrace$, with
   $\lambda_\beta$ re-chosen by CV for that set. Drop the atom whose removal gives the lowest score.
3. Repeat down to one non-constant atom. This gives a path of nested sets $\mathcal A_0 \supset \mathcal A_1 \supset \dots$,
   each with a pooled score $\mathrm{CV}_j$ and a standard error $\mathrm{SE}_j$ (standard deviation of its 5 block
   scores divided by $\sqrt 5$).
4. Let $j^\ast$ minimise $\mathrm{CV}_j$. Select the **smallest** set with $\mathrm{CV}_j \le \mathrm{CV}_{j^\ast} + \mathrm{SE}_{j^\ast}$.
5. Refit $b$ on the selected set with all time steps and its $\lambda_\beta$. Call the result $b_{\rm 2s}$.

All later variants keep this atom set fixed.

### Step 2 (hard constraint): fit the operator through the rollout

The state stays at $\bar\alpha$, but the coefficients are chosen so that the rollout from its first column matches all
later columns:

$$\min_{b_{\mathcal A}}\ \sum_{n=1}^{N_T-1} \lVert K\bar\alpha_n - K\,\Phi^n(\bar\alpha_0; b) \rVert^2
 + \lambda_\beta \sum_{m \in \mathcal A} \beta_m^\top K_m \beta_m, \qquad \lambda_\beta = 10^{-6} .$$

This is nonlinear in $b$. It is solved as a least-squares problem (the penalty enters as
$\sqrt{\lambda_\beta}\, L_{\mathcal A}^\top b$ with $K_{\rm reg} + 10^{-10} I = L_{\mathcal A} L_{\mathcal A}^\top$) by
trust-region Gauss-Newton (`scipy.optimize.least_squares`, `trf`), with the exact Jacobian from forward-mode autodiff
through the rollout, started from $b_{\rm 2s}$. A rollout that blows up is turned into a residual of $10^6$ so that
the step is rejected.

## 6. One-step methods

State and coefficients are fitted together; the atom set, standardisation and inducing points stay frozen.

### Soft constraint

Unknowns: all columns $\alpha_0, \dots, \alpha_{N_T-1}$ and $b_{\mathcal A}$.

$$J(\alpha, b) = \lambda_d \sum_{(n,k) \in \Omega} \big(y_{nk} - (K\alpha_n)_k\big)^2
 + \lambda_{\rm dyn} \sum_{n} \lVert e(\alpha_{n+1}, \alpha_n; b) \rVert^2
 + \lambda_\alpha \sum_n \alpha_n^\top K \alpha_n + \lambda_\beta \sum_{m} \beta_m^\top K_m \beta_m
 + \frac{\lambda_{\ddot a}}{\Delta t^4} \sum_n \lVert \alpha_{n+1} - 2\alpha_n + \alpha_{n-1} \rVert_K^2 .$$

All $N_X$ rows of $e$ are used, so the boundary rows hold the boundary values fixed. Minimised by L-BFGS-B in
$(w, b_{\mathcal A})$ with the exact gradient, started from the two-step solution $(\bar\alpha, b_{\rm 2s})$; the loss
is divided by its starting value so the tolerances are scale-free.

*Weights.* $\lambda_d = 1$, $\lambda_\alpha = \lambda_u$, $\lambda_\beta = 10^{-8}$. By 5-fold CV on held-out interior
observations (the Step-1 folds), in sequence: first $\lambda_{\rm dyn}$ over $\lbrace 0\rbrace \cup \lbrace 10^{-4}, \dots, 10^{2}\rbrace$
(half decades) with $\lambda_{\ddot a} = 0$, then $\lambda_{\ddot a}$ over
$\lbrace 0, 10^{-10}, 10^{-9}, 10^{-8}, 10^{-7}, 10^{-6}\rbrace$ plus the Step-1 value.

### Hard constraint

Unknowns: $\alpha_0$ and $b_{\mathcal A}$ only. Every later column is $\alpha_n = \Phi^n(\alpha_0; b)$, so the dynamics
hold exactly and the $\lambda_{\rm dyn}$ term disappears:

$$J(\alpha_0, b) = \lambda_d \sum_{(n,k)\in\Omega} \big(y_{nk} - (K\alpha_n)_k\big)^2
 + \lambda_\alpha\, \alpha_0^\top K \alpha_0 + \lambda_\beta \sum_m \beta_m^\top K_m \beta_m .$$

Solved like the two-step hard fit (least squares, Jacobian through the rollout), started from the two-step soft
solution $(\bar\alpha_0, b_{\rm 2s})$.

The package penalises only the free variables $\alpha_0$ and $b$. The notebook (where this cell runs only for
experiment 3A) differs in three ways: it penalises $\lambda_\alpha \sum_n \alpha_n^\top K \alpha_n$ over all
rolled-out columns, which fails when $K$ is ill-conditioned (section 9); it adds the time prior
$\frac{\lambda_{\ddot a}}{\Delta t^4} \sum_n \lVert \alpha_{n+1} - 2\alpha_n + \alpha_{n-1} \rVert_K^2$ on the rolled-out
columns, with $\lambda_{\ddot a}$ by the same CV; and it starts from the one-step soft solution.

## 7. Summary and defaults

```
input:  grid (t_n, x_k), observations y on Omega, spatial kernel k
1. basis      K, D_0..D_3 from k;  K + 1e-8 I = L L^T
2. Step 1     alpha_bar = ridge fit of all columns with the time prior; lam_addot by 5-fold CV
3. library    features g(s), standardised on alpha_bar; atoms = features x kernels + constant
4. Step 2     backward elimination over atoms, scored by time-block CV (lam_beta by CV per set)
              -> one-standard-error rule -> active set A, coefficients b_2s (closed form)
5. optional   two-step hard:  b_2h = fit through the rollout from alpha_bar_0
              one-step soft:  (alpha_1s, b_1s) = joint fit, lam_dyn and lam_addot by CV
              one-step hard:  (alpha_0, b_1h), all columns = rollout, from (alpha_bar_0, b_2s)
output: P_hat(s) = Psi_A(s) b;  rollout alpha_n = Phi^n(K^-1 u_0; b)
```

| Symbol | Value | Used in |
|---|---|---|
| $\theta$, Newton iterations | $1/2$, 6 | time step |
| nugget in $K + \epsilon I = L L^\top$ | $10^{-8}$ | whitening |
| $\lambda_u$ | $10^{-6}$ | Step 1 (and $\lambda_\alpha$ in the one-step losses; on $\alpha_0$ only in the hard one) |
| $\lambda_{\ddot a}$ | CV, $\lbrace 0\rbrace \cup 10^{-12 \dots -1}$ | Step 1 |
| $J$ inducing points | 15 | library |
| $\lambda_{\rm dyn}$ | 1 | Step 2 soft |
| $\lambda_\beta$ | CV, $10^{-12 \dots 2}$ | Step 2 soft and selection |
| $\lambda_\beta$ | $10^{-6}$ | Step 2 hard |
| $\lambda_d$, $\lambda_\beta$ | 1, $10^{-8}$ | one-step losses |
| $\lambda_{\rm dyn}$ | CV, $\lbrace 0\rbrace \cup 10^{-4 \dots 2}$ | one-step soft |
| folds | 5 random folds of interior observations (state); 5 contiguous blocks of time steps (operator) | CV |

## 8. The package

`kernel_timestepping` implements sections 2 to 6. Kernels and kernel functionals come from
[JaxKernels](https://github.com/AHsu98/JaxKernels).

| Module | Contents | Notebook cell |
|---|---|---|
| `spacetime_grid.py` | `SpaceTimeGrid`, `NodalObservations`, `sample_nodal_observations`, `blended_chebyshev_nodes`, `interior_folds`, `time_blocks` | data cell |
| `state_basis.py` | `StateBasis` ($K$, $D_q$, whitening, jets) | "Pick state kernel" |
| `state_lengthscale.py` | `state_lengthscale` (the marginal-likelihood fit of section 2) | "Pick state kernel" |
| `kmd_library.py` | `Feature`, `Atom`, `AtomLibrary`, `build_library` | "Operator KMD library" |
| `theta_stepper.py` | `ThetaStepper` (residual, Newton step, rollout) | `make_stepper` |
| `two_step_state_fit.py` | `fit_state`, `ridge_svd` | "State fit" |
| `two_step_operator_soft.py` | `OperatorRegression` | "Operator fit" |
| `atom_selection.py` | `backward_elimination`, `one_standard_error`, `select_atoms` | "Operator fit" |
| `two_step_operator_hard.py` | `fit_two_step_operator_hard` | "2-step hard" |
| `one_step_soft.py` | `OneStepWeights`, `fit_one_step_soft` | "1-step soft" |
| `one_step_hard.py` | `fit_one_step_hard` | "1-step hard" |
| `cross_validation.py` | `heldout_rmse`, `grid_search` | CV loops |
| `jax_optimizers.py` | `least_squares_jax`, `lbfgs_jax` | |
| `pipeline.py` | `KMDTimeStepper`: the four methods as one object, `fit(obs, method=...)` | |

The reference solver and spline of its solution, for experiments and tests only, are in
[`problems/burgers_reference.py`](../problems/burgers_reference.py), outside the package.

```python
from jaxkernels import GaussianRBFKernel
import kernel_timestepping as kt

grid = kt.SpaceTimeGrid(x=kt.blended_chebyshev_nodes(31), dt=0.04, n_t=26)
obs = kt.NodalObservations.from_points(grid, tx_obs, u_obs)          # (t, x) coordinates and values
model = kt.KMDTimeStepper(GaussianRBFKernel(lengthscale=kt.state_lengthscale(grid, obs)), grid)
model.fit(obs, method="two_step_soft")                               # or two_step_hard, one_step_soft, one_step_hard
print(model.summary())                                               # selected atoms, coefficients
u = model.rollout(u0_nodes, n_steps=50)                              # (51, 31) nodal values
```

[`experiments/burgers_walkthrough.ipynb`](../experiments/burgers_walkthrough.ipynb) goes through the stages one function at a
time. Install with `pip install -e ".[examples,test]"`; run the tests with `pytest`.

**How JaxKernels is used.**

* The state kernel is any JaxKernels kernel on scalars. $D_q$ is `make_block(kernel, derivative_op(q), eval_k)`.
* Every atom kernel is a JaxKernels kernel on scalars, and every atom is "kernel + inducing points". The linear atom
  is `LinearKernel` with the single inducing point 1 ($\phi(z) = z$, $K_m = 1$), the constant atom is
  `ConstantKernel`, the others are `GaussianRBFKernel(lengthscale=1)` and `MaternKernel(p=1, lengthscale=1)`. Other
  kernels can be passed to `build_library`.
* The lengthscale fit uses `objectives.neg_log_marginal_likelihood` and the log-coordinate helpers in `hyper`.
* Fold construction uses `objectives.kfold_indices`.

**Where the package differs from the notebook.**

* Coefficient vectors hold the active atoms only (the notebook keeps all 187 entries and zeroes the inactive ones).
  Trajectories are stored as `(N_T, N_X)` arrays, row $n$ = $\alpha_n$.
* The Matérn atom uses JaxKernels' exact derivatives at $z = \zeta$; the notebook adds $10^{-30}$ under a square root.
* In the selection, the normal equations of a CV block are formed once and reused for all $\lambda_\beta$. The
  arithmetic is unchanged.
* The lengthscale is an input. `state_lengthscale` minimises the same objective as the `keql_tools` routine the
  notebook calls, and agrees with it to 4 digits on the notebook's data.
* Not ported: the Kernel Equation Learning comparison (its two-step and one-step fits, the degree-2 feature map
  `psi`, the `R_filter` figures) and the plotting code.

## 9. Checks and observations

### The package against the notebook's printed output

Run on the notebook's own data for experiment 3B (shock; $\kappa = 0.001$, $\alpha = 5$, 60 interior observations),
with the lengthscale the committed run printed ($0.75/\sqrt{197.6221}$):

| Quantity | Notebook | Package |
|---|---|---|
| $\lambda_{\ddot a}$ chosen; CV error; per-column CV error | 1.0e-10; 6.23e-02; 1.27e-01 | same |
| largest misfit at the observations | 6.8e-05 | 6.8e-05 |
| relative error of $u$, $u_x$, $u_{xx}$ (Step 1) | 0.468, 1.02, 1.00 | same |
| lowest CV on the elimination path | 2.392e-02 ± 6.3e-03 at 2 atoms | same |
| one-standard-error choice | `linear[u]`, CV 2.444e-02, $\lambda_\beta$ = 1e-12 | same |
| two-step soft coefficients (const, `linear[u]`) | 6.38312e-02, −3.60658e-01 | 6.38312e-02, −3.60658e-01 |
| two-step hard coefficients; loss; evaluations | 7.77597e-02, −5.03007e-01; 1.410; 7 | 7.77597e-02, −5.03007e-01; 1.410; 7 |
| one-step soft coefficients at $\lambda_{\rm dyn}$ = 1e-4, $\lambda_{\ddot a}$ = 1e-10 | 1.23263e-01, −3.38687e-01 | 1.23263e-01, −3.38688e-01 |

Not compared: the one-step CV loops on 3B (the weights above are the ones the notebook chose), and the one-step hard
fit, for which the notebook has no saved output.

Experiment 3B is the case where the method does not recover the equation, in the notebook as in the package: one
atom, `linear[u]`, is selected, and the notebook reports that even the true equation cannot be rolled out with this
stepper on the 26 × 31 grid ("stepper floor ... nan").

### The package on the notebook's 3A data

The notebook has no saved output for experiment 3A ($\kappa = 0.01$, $\alpha = 1$, smooth initial condition, 100
interior observations, $\ell_x = 0.0755$). The package gives:

| Variant | Fitted operator (atoms selected: const, `linear[u_xx]`, `linear[u*u_x]`) | Operator error | Rollout error |
|---|---|---|---|
| two-step soft | −0.0043 + 0.00940 $u_{xx}$ + 0.943 $u u_x$ | 0.060 | 0.025 |
| two-step hard | −0.0001 + 0.00998 $u_{xx}$ + 1.014 $u u_x$ | 0.015 | 0.0037 |
| one-step soft ($\lambda_{\rm dyn}$ = 3.2e-2, $\lambda_{\ddot a}$ = 0 by CV) | −0.0001 + 0.00998 $u_{xx}$ + 0.9974 $u u_x$ | 0.0020 | 0.00092 |
| one-step hard ($\lambda_{\ddot a}$ = 0) | −0.00001 + 0.0099993 $u_{xx}$ + 0.9986 $u u_x$ | 0.0013 | 0.00065 |
| true equation in the same stepper | | | 0.00068 |

Operator error: $\hat P$ at the true features against the true $u_t$ on a 100 × 100 grid, relative RMS. Rollout
error: relative $L^2$ error at the nodes of the rollout from the true initial column. The elimination path has its
lowest CV score at 5 atoms; the one-standard-error rule reduces it to the two true terms.

### Things to know when using it

* **The selected atoms depend on which nodes are observed.** In the example notebook (five draws of the 100 interior
  observations), all five find $u_{xx}$ and $u u_x$ as linear atoms; two also keep one RBF or Matérn atom on $u_{xxx}$
  and roll out less accurately (3 to 5 % against 1 to 1.5 %). With a different initial condition
  ($0.3 \sin \pi x + 0.2 \sin 2\pi x - 0.1 \sin 3\pi x$), one draw of three selected an RBF atom on $u u_x$ in place of
  the linear one, and its rollout diverged.
* **The notebook's one-step hard loss is sensitive to the conditioning of $K$; the package's is not.** The notebook
  penalises $\lambda_\alpha \sum_n \alpha_n^\top K \alpha_n$ on rolled-out columns, and every step of a rollout solves a
  linear system with a matrix close to $K$.
  $\mathrm{cond}(K)$ is 4e5 at $\ell_x = 0.053$ (3B), 8e8 at 0.0755 (3A) and 4e14 at 0.109 (example notebook). In the
  example, the rollout of the one-step soft solution has data misfit 1.2e-6 but $\lambda_\alpha \times$ norm = 3.2e3 at
  $\lambda_\alpha = 10^{-6}$; minimising the hard loss from there moves the coefficient of $u u_x$ from 0.99 to 0.50.
  Lowering $\lambda_\alpha$ to $10^{-12}$ still leaves it at 0.87. On the 3A data the same fit works (table above).
  Penalising $\alpha_0$ only (the package's `fit_one_step_hard`) gives 0.9982 for the coefficient of $u u_x$, 0.01000
  for $u_{xx}$ and a rollout error of 0.03 % in the example, from the two-step soft fit with the default weights; the
  result is the same for $\lambda_\alpha$ on $\alpha_0$ from $10^{-10}$ to $10^{-6}$. Over the five observation draws
  of the example it improves on the two-step soft rollout error in all five (0.03-0.04 % on the three draws that
  select only the true atoms; 1.2 % and 1.0 % on the two that keep an extra atom on $u_{xxx}$, where the optimiser stops after
  3 to 4 evaluations).
  The soft variants are not affected: their states are penalised fits, not rollouts.
* **The two-step hard fit can stall with kernel atoms on $u_{xxx}$.** For one of the draws above, the optimiser
  stopped after 11 evaluations with unchanged coefficients (step-size tolerance reached, gradient not small).
  `KMDTimeStepper.refit_result_` keeps the optimiser result so this can be checked.
* **The one-step soft fit often ends at the iteration cap** (20 000 L-BFGS iterations). On the 3A data it converged
  after about 11 500; in the example it reached the cap, and running it to convergence (about 29 400 iterations)
  changed the coefficients in the sixth digit.
