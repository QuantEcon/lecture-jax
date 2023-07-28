---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Inventory Management Model

```{include} _admonition/gpu.md
```

```{code-cell} ipython3
!pip install quantecon
```

```{code-cell} ipython3
import quantecon as qe
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
```

Let's check the GPU we are running

```{code-cell} ipython3
!nvidia-smi
```

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

```{code-cell} ipython3
# NamedTuple Model
Model = namedtuple("Model", ("K", "c", "κ", "p", "z_vals", "Q"))
```

```{code-cell} ipython3
@jax.jit
def demand_pdf(p, d):
    return (1 - p)**d * p
```

```{code-cell} ipython3
def create_sdd_inventory_model(
        ρ=0.98, ν=0.002, n_z=20, b=0.97,   # Z state parameters
        K=40, c=0.2, κ=0.8, p=0.6):        # firm and demand parameters
    mc = qe.tauchen(n_z, ρ, ν)
    z_vals, Q = jnp.array(mc.state_values + b), jnp.array(mc.P)
    # rL = jnp.max(jnp.abs(jnp.linalg.eigvals(z_vals * Q)))
    # assert rL < 1, "Error: r(L) >= 1."    # check r(L) < 1
    return Model(K=K, c=c, κ=κ, p=p, z_vals=z_vals, Q=Q)
```

```{code-cell} ipython3
Model_K = 40
D_MAX = 101
```

```{code-cell} ipython3
@jax.jit
def B(x, i_z, a, v, model):
    """
    The function B(x, z, a, v) = r(x, a) + β(z) Σ_x′ v(x′) P(x, a, x′).
    """
    K, c, κ, p, z_vals, Q = model
    z = z_vals[i_z]
    d_range = jnp.arange(D_MAX)
    demand = demand_pdf(p, d_range)
    _tmp = jnp.minimum(x, d_range)*demand
    reward = jnp.sum(_tmp) - c * a - κ * (a > 0)
    _tmp = jnp.sum(v[jnp.maximum(x - d_range, 0) + a].T * demand, axis=1)
    cv = jnp.sum(_tmp*Q[i_z])
    return reward + z * cv
```

```{code-cell} ipython3
B_vec1 = jax.vmap(B, in_axes=(None, None, 0, None, None))
```

```{code-cell} ipython3
@jax.jit
def B2(x, i_z, v, model):
    """
    The function B(x, z, a, v) = r(x, a) + β(z) Σ_x′ v(x′) P(x, a, x′).
    """
    K, c, κ, p, z_vals, Q = model
    a_range = jnp.arange(Model_K)
    res = B_vec1(x, i_z, a_range, v, model)
    return jnp.where(a_range < Model_K - x + 1, res, -jnp.inf)
```

```{code-cell} ipython3
B_vec2 = jax.vmap(B2, in_axes=(None, 0, None, None))
B_vec3 = jax.vmap(B_vec2, in_axes=(0, None, None, None))
```

```{code-cell} ipython3
@jax.jit
def T(v, model):
    """The Bellman operator."""
    K, c, κ, p, z_vals, Q = model
    i_z_range = jnp.arange(len(z_vals))
    x_range = jnp.arange(Model_K + 1)
    res = B_vec3(x_range, i_z_range, v, model)
    return jnp.max(res,axis=2)
```

```{code-cell} ipython3
@jax.jit
def get_greedy(v, model):
    """Get a v-greedy policy.  Returns a zero-based array."""
    K, c, κ, p, z_vals, Q = model
    i_z_range = jnp.arange(len(z_vals))
    x_range = jnp.arange(Model_K + 1)
    res = B_vec3(x_range, i_z_range, v, model)
    return jnp.argmax(res,axis=2)
```

```{code-cell} ipython3
def successive_approx(T,                     # Operator (callable)
                      x_0,                   # Initial condition
                      tolerance=1e-6,        # Error tolerance
                      max_iter=10_000,       # Max iteration bound
                      print_step=25,         # Print at multiples
                      verbose=False):
    x = x_0
    error = tolerance + 1
    k = 1
    while error > tolerance and k <= max_iter:
        x_new = T(x)
        error = jnp.max(jnp.abs(x_new - x))
        if verbose and k % print_step == 0:
            print(f"Completed iteration {k} with error {error}.")
        x = x_new
        k += 1
    if error > tolerance:
        print(f"Warning: Iteration hit upper bound {max_iter}.")
    elif verbose:
        print(f"Terminated successfully in {k} iterations.")
    return x
```

```{code-cell} ipython3
def solve_inventory_model(v_init, model):
    """Use successive_approx to get v_star and then compute greedy."""
    v_star = successive_approx(lambda v: T(v, model), v_init, verbose=True)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
```

```{code-cell} ipython3
model = create_sdd_inventory_model()
K, c, κ, p, z_vals, Q = model
n_z = len(z_vals)
v_init = jnp.zeros((Model_K + 1, n_z), dtype=float)
```

```{code-cell} ipython3
%time v_star, σ_star = solve_inventory_model(v_init, model)
```

```{code-cell} ipython3
z_mc = qe.MarkovChain(Q, z_vals)
```

```{code-cell} ipython3
def sim_inventories(ts_length, X_init=0):
    """Simulate given the optimal policy."""
    global p, z_mc
    i_z = z_mc.simulate_indices(ts_length, init=1)
    X = np.zeros(ts_length, dtype=np.int32)
    X[0] = X_init
    rand = np.random.default_rng().geometric(p=p, size=ts_length-1) - 1
    for t in range(ts_length-1):
        X[t+1] = np.maximum(X[t] - rand[t], 0) + σ_star[X[t], i_z[t]]
    return X, z_vals[i_z]
```

```{code-cell} ipython3
def plot_ts(ts_length=400, fontsize=10):
    X, Z = sim_inventories(ts_length)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))

    ax = axes[0]
    ax.plot(X, label=r"$X_t$", alpha=0.7)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel("inventory", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, np.max(X)+3)

    # calculate interest rate from discount factors
    r = (1 / Z) - 1

    ax = axes[1]
    ax.plot(r, label=r"$r_t$", alpha=0.7)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel("interest rate", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
plot_ts()
```
