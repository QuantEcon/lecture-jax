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

# Optimal Savings

```{include} _admonition/gpu.md
```

In addition to what’s in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

We will use the following imports:

```{code-cell} ipython3
import quantecon as qe
import jax
import jax.numpy as jnp
from collections import namedtuple
import matplotlib.pyplot as plt
import time
```

Let's check the GPU we are running

```{code-cell} ipython3
!nvidia-smi
```

Use 64 bit floats with JAX in order to match NumPy code
- By default, JAX uses 32-bit datatypes.
- By default, NumPy uses 64-bit datatypes.

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

## Overview

We consider an optimal savings problem with CRRA utility and budget constraint

$$ W_{t+1} + C_t \leq R W_t + Y_t $$

We assume that labor income $(Y_t)$ is a discretized AR(1) process.

The right-hand side of the Bellman equation is

$$   B((w, y), w', v) = u(Rw + y - w') + β \sum_{y'} v(w', y') Q(y, y'). $$

where

$$   u(c) = \frac{c^{1-\gamma}}{1-\gamma} $$

+++

We use successive approximation for VFI.

```{code-cell} ipython3
:load: _static/lecture_specific/successive_approx.py
```

## Model primitives

Here’s a `namedtuple` definition for storing parameters and grids.

```{code-cell} ipython3
Model = namedtuple('Model',
                    ('β', 'R', 'γ', 'w_grid', 'y_grid', 'Q'))
```

```{code-cell} ipython3
def create_consumption_model(R=1.01,                    # Gross interest rate
                             β=0.98,                    # Discount factor
                             γ=2.5,                     # CRRA parameter
                             w_min=0.01,                # Min wealth
                             w_max=5.0,                 # Max wealth
                             w_size=150,                # Grid side
                             ρ=0.9, ν=0.1, y_size=100): # Income parameters
    """
    A function that takes in parameters and returns an instance of Model that
    contains data for the optimal savings problem.
    """
    w_grid = jnp.linspace(w_min, w_max, w_size)
    mc = qe.tauchen(n=y_size, rho=ρ, sigma=ν)
    y_grid, Q = jnp.exp(mc.state_values), mc.P
    return Model(β=β, R=R, γ=γ, w_grid=w_grid, y_grid=y_grid, Q=Q)
```

```{code-cell} ipython3
def create_consumption_model_jax(R=1.01,                # Gross interest rate
                             β=0.98,                    # Discount factor
                             γ=2.5,                     # CRRA parameter
                             w_min=0.01,                # Min wealth
                             w_max=5.0,                 # Max wealth
                             w_size=150,                # Grid side
                             ρ=0.9, ν=0.1, y_size=100): # Income parameters
    """
    A function that takes in parameters and returns a JAX-compatible version of
    Model that contains data for the optimal savings problem.
    """
    w_grid = jnp.linspace(w_min, w_max, w_size)
    mc = qe.tauchen(n=y_size, rho=ρ, sigma=ν)
    y_grid, Q = jnp.exp(mc.state_values), mc.P
    β, R, γ = jax.device_put([β, R, γ])
    w_grid, y_grid, Q = tuple(map(jax.device_put, [w_grid, y_grid, Q]))
    sizes = w_size, y_size
    return (β, R, γ), sizes, (w_grid, y_grid, Q)
```

Here's the right hand side of the Bellman equation:

```{code-cell} ipython3
@jax.jit
def compute_c(R, w, y, wp):
    return R * w + y - wp

compute_c_vec1 = jax.vmap(compute_c, in_axes=(None, None, None, 0))
compute_c_vec2 = jax.vmap(compute_c_vec1, in_axes=(None, None, 0, None))
compute_c_vec3 = jax.vmap(compute_c_vec2, in_axes=(None, 0, None, None))
```


```{code-cell} ipython3
def B(v, constants, sizes, arrays):
    """
    A vectorized version of the right-hand side of the Bellman equation
    (before maximization), which is a 3D array representing

        B(w, y, w′) = u(Rw + y - w′) + β Σ_y′ v(w′, y′) Q(y, y′)

    for all (w, y, w′).
    """

    # Unpack
    β, R, γ = constants
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    # Compute current rewards r(w, y, wp) as array r[i, j, ip]
    c = compute_c_vec3(R, w_grid, y_grid, w_grid)

    # Calculate continuation rewards at all combinations of (w, y, wp)
    v = jnp.reshape(v, (1, 1, w_size, y_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, y_size, 1, y_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    EV = jnp.sum(v * Q, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return jnp.where(c > 0, c**(1-γ)/(1-γ) + β * EV, -jnp.inf)
```

## Operators

Now we define the policy operator $T_\sigma$

```{code-cell} ipython3
def compute_r_σ(σ, constants, sizes, arrays):
    """
    Compute the array r_σ[i, j] = r[i, j, σ[i, j]], which gives current
    rewards given policy σ.
    """

    # Unpack model
    β, R, γ = constants
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    # Compute r_σ[i, j]
    w = jnp.reshape(w_grid, (w_size, 1))
    y = jnp.reshape(y_grid, (1, y_size))
    wp = w_grid[σ]
    c = R * w + y - wp
    r_σ = c**(1-γ)/(1-γ)

    return r_σ
```

```{code-cell} ipython3
def T_σ(v, σ, constants, sizes, arrays):
    "The σ-policy operator."

    # Unpack model
    β, R, γ = constants
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    r_σ = compute_r_σ(σ, constants, sizes, arrays)

    # Compute the array v[σ[i, j], jp]
    yp_idx = jnp.arange(y_size)
    yp_idx = jnp.reshape(yp_idx, (1, 1, y_size))
    σ = jnp.reshape(σ, (w_size, y_size, 1))
    V = v[σ, yp_idx]

    # Convert Q[j, jp] to Q[i, j, jp]
    Q = jnp.reshape(Q, (1, y_size, y_size))

    # Calculate the expected sum Σ_jp v[σ[i, j], jp] * Q[i, j, jp]
    Ev = jnp.sum(V * Q, axis=2)

    return r_σ + β * Ev
```

and the Bellman operator $T$

```{code-cell} ipython3
def T(v, constants, sizes, arrays):
    "The Bellman operator."
    return jnp.max(B(v, constants, sizes, arrays), axis=2)
```

The next function computes a $v$-greedy policy given $v$

```{code-cell} ipython3
def get_greedy(v, constants, sizes, arrays):
    "Computes a v-greedy policy, returned as a set of indices."
    return jnp.argmax(B(v, constants, sizes, arrays), axis=2)
```

The function below computes the value $v_\sigma$ of following policy $\sigma$.

The basic problem is to solve the linear system

$$ v(w,y ) = u(Rw + y - \sigma(w, y)) + β \sum_{y'} v(\sigma(w, y), y') Q(y, y) $$

for $v$.

It turns out to be helpful to rewrite this as

$$ v(w,y) = r(w, y, \sigma(w, y)) + β \sum_{w', y'} v(w', y') P_\sigma(w, y, w', y') $$

where $P_\sigma(w, y, w', y') = 1\{w' = \sigma(w, y)\} Q(y, y')$.

We want to write this as $v = r_\sigma + P_\sigma v$ and then solve for $v$

Note, however,

* $v$ is a 2 index array, rather than a single vector.
* $P_\sigma$ has four indices rather than 2

The code below

1. reshapes $v$ and $r_\sigma$ to 1D arrays and $P_\sigma$ to a matrix
2. solves the linear system
3. converts back to multi-index arrays.

```{code-cell} ipython3
def R_σ(v, σ, constants, sizes, arrays):
    """
    The value v_σ of a policy σ is defined as

        v_σ = (I - β P_σ)^{-1} r_σ

    Here we set up the linear map v -> R_σ v, where R_σ := I - β P_σ.

    In the consumption problem, this map can be expressed as

        (R_σ v)(w, y) = v(w, y) - β Σ_y′ v(σ(w, y), y′) Q(y, y′)

    Defining the map as above works in a more intuitive multi-index setting
    (e.g. working with v[i, j] rather than flattening v to a one-dimensional
    array) and avoids instantiating the large matrix P_σ.

    """

    β, R, γ = constants
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    # Set up the array v[σ[i, j], jp]
    zp_idx = jnp.arange(y_size)
    zp_idx = jnp.reshape(zp_idx, (1, 1, y_size))
    σ = jnp.reshape(σ, (w_size, y_size, 1))
    V = v[σ, zp_idx]

    # Expand Q[j, jp] to Q[i, j, jp]
    Q = jnp.reshape(Q, (1, y_size, y_size))

    # Compute and return v[i, j] - β Σ_jp v[σ[i, j], jp] * Q[j, jp]
    return v - β * jnp.sum(V * Q, axis=2)
```

```{code-cell} ipython3
def get_value(σ, constants, sizes, arrays):
    "Get the value v_σ of policy σ by inverting the linear map R_σ."

    # Unpack
    β, R, γ = constants
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    r_σ = compute_r_σ(σ, constants, sizes, arrays)

    # Reduce R_σ to a function in v
    partial_R_σ = lambda v: R_σ(v, σ, constants, sizes, arrays)

    return jax.scipy.sparse.linalg.bicgstab(partial_R_σ, r_σ)[0]
```

## JIT compiled versions

```{code-cell} ipython3
B = jax.jit(B, static_argnums=(2,))
compute_r_σ = jax.jit(compute_r_σ, static_argnums=(2,))
T = jax.jit(T, static_argnums=(2,))
get_greedy = jax.jit(get_greedy, static_argnums=(2,))
get_value = jax.jit(get_value, static_argnums=(2,))
T_σ = jax.jit(T_σ, static_argnums=(3,))
R_σ = jax.jit(R_σ, static_argnums=(3,))
```

## Solvers

Now we define the solvers, which implement VFI, HPI and OPI.

```{code-cell} ipython3
:load: _static/lecture_specific/vfi.py
```

```{code-cell} ipython3
:load: _static/lecture_specific/hpi.py
```

```{code-cell} ipython3
:load: _static/lecture_specific/opi.py
```

## Plots

Create a JAX model for consumption, perform policy iteration, and plot the resulting optimal policy function.

```{code-cell} ipython3
fontsize = 12
model = create_consumption_model_jax()
# Unpack
constants, sizes, arrays = model
β, R, γ = constants
w_size, y_size = sizes
w_grid, y_grid, Q = arrays
```

```{code-cell} ipython3
σ_star = policy_iteration(model)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(w_grid, w_grid, "k--", label="45")
ax.plot(w_grid, w_grid[σ_star[:, 1]], label="$\\sigma^*(\cdot, y_1)$")
ax.plot(w_grid, w_grid[σ_star[:, -1]], label="$\\sigma^*(\cdot, y_N)$")
ax.legend(fontsize=fontsize)
plt.show()
```

## Tests

Here's a quick test of the timing of each solver.

```{code-cell} ipython3
model = create_consumption_model_jax()
```

```{code-cell} ipython3
print("Starting HPI.")
start_time = time.time()
out = policy_iteration(model)
elapsed = time.time() - start_time
print(f"HPI completed in {elapsed} seconds.")
```

```{code-cell} ipython3
print("Starting VFI.")
start_time = time.time()
out = value_iteration(model)
elapsed = time.time() - start_time
print(f"VFI(jax not in succ) completed in {elapsed} seconds.")
```

```{code-cell} ipython3
print("Starting OPI.")
start_time = time.time()
out = optimistic_policy_iteration(model, m=100)
elapsed = time.time() - start_time
print(f"OPI completed in {elapsed} seconds.")
```

```{code-cell} ipython3
def run_algorithm(algorithm, model, **kwargs):
    start_time = time.time()
    result = algorithm(model, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{algorithm.__name__} completed in {elapsed_time:.2f} seconds.")
    return result, elapsed_time
```

```{code-cell} ipython3
model = create_consumption_model_jax()
σ_pi, pi_time = run_algorithm(policy_iteration, model)
σ_vfi, vfi_time = run_algorithm(value_iteration, model, tol=1e-5)

m_vals = range(5, 3000, 100)
opi_times = []
for m in m_vals:
    σ_opi, opi_time = run_algorithm(optimistic_policy_iteration, model, m=m, tol=1e-5)
    opi_times.append(opi_time)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(m_vals, jnp.full(len(m_vals), pi_time), lw=2, label="Howard policy iteration")
ax.plot(m_vals, jnp.full(len(m_vals), vfi_time), lw=2, label="value function iteration")
ax.plot(m_vals, opi_times, lw=2, label="optimistic policy iteration")
ax.legend(fontsize=fontsize, frameon=False)
ax.set_xlabel("$m$", fontsize=fontsize)
ax.set_ylabel("time", fontsize=fontsize)
plt.show()
```
