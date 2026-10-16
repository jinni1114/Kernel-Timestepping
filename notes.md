We want to recover the linear kernels, but we keep just recovering matern, or other kernels

First thing thrown away is u u_x
Then throw away matern u_xx

Forward greedy tends to keep the linear ones. 

Choose kernels here by marginal likelihood?
Leave one out cross validation? fast and can help a little bit

You're doing leave one chunk out (for picking kernels for P?)

Look at ways to improve the initial fit?

1. Derive a weak form? This would be hard in general,

   $$
   u_{x} u_{xxx}
   $$

   Maybe replace collocation loss with galerkin, or weaker sobolev matching.

   $$
   \min_{P} \sum_{t_k} \|P(u_{t_k}) - \partial_t u_{t_k}\|_{L^2(\mathcal{X})}
   $$

Maybe we can target something like

$$
\min_{P} \sum_{t_k} \|P(u_{t_k}) - \partial_t u_{t_k}\|_{H^{-1/2}(\mathcal{X})}
$$

, using the kernel representation. This might be hard computationally.

Alternately, do Galerkin,

$$
\min_{P} \sum_{t_k} \sum_{r} |\langle \phi_{r},P(u_{t_k}) - \partial_t u_{t_k}\rangle|^2
$$

2. Iteratively re-apply feature selection. Afterwards, take all of the features again, redo feature selection. If we're doing this, then don't fully throw away bad features, leave $\epsilon k_i$ if we're "getting rid of" $k_i$. 
3. Soft downfeatures, (look into connection between KMD, and iteratively reweighted least squares on the block lasso). Look into iteratively reweighted least squares for lasso. Francis bach blog post (primal dual interpretation?). Not clear how to do this through cross validation, will take thinking, but maybe can have good idea. 
4. This would be computationally too expensive, 2^p possible feature combinations. Do the 1-step method, with with all fixed different possible feature combinations, and pick the best one based on cross validation. 