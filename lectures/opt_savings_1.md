---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Optimal Savings I: Value Function Iteration

```{include} _admonition/gpu.md
```

In addition to JAX and Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

We will use the following imports:

```{code-cell} ipython3
import quantecon as qe
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import namedtuple
from time import time
from typing import NamedTuple
from functools import partial
```

Let's check the GPU we are running

```{code-cell} ipython3
!nvidia-smi
```

We'll use 64 bit floats to gain extra precision.

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

## Overview

We consider an optimal savings problem with CRRA utility and budget constraint

$$ 
W_{t+1} + C_t \leq R W_t + Y_t 
$$

where

* $C_t$ is consumption and $C_t \geq 0$,
* $W_t$ is wealth and $W_t \geq 0$,
* $R > 0$ is a gross rate of return, and
* $(Y_t)$ is labor income.

We assume below that labor income is a discretized AR(1) process.

The Bellman equation is

$$   
    v(w) = \max_{0 \leq w' \leq Rw + y}
    \left\{
        u(Rw + y - w') + β \sum_{y'} v(w', y') Q(y, y') 
    \right\}
$$

where

$$
    u(c) = \frac{c^{1-\gamma}}{1-\gamma} 
$$

In the code we use the function

$$   
    B((w, y), w', v) = u(Rw + y - w') + β \sum_{y'} v(w', y') Q(y, y'). 
$$

the encapsulate the right hand side of the Bellman equation.



## Starting with NumPy

Let's start with a standard NumPy version running on the CPU.

Starting with this traditional approach will allow us to record the speed gain
associated with switching to JAX.

(NumPy operations are similar to MATLAB operations, so this also serves as a
rough comparison with MATLAB.)



### Functions and operators

The following class contains default parameters and arrays
storing the key computational components of the model.

```{code-cell} ipython3
class Model(NamedTuple):
    β: float
    R: float
    γ: float
    w_grid: jnp.ndarray
    y_grid: jnp.ndarray
    Q: jnp.ndarray
```

```{code-cell} ipython3
def create_consumption_model(R=1.01,          # Gross interest rate
                             β=0.98,          # Discount factor
                             γ=2,             # CRRA parameter
                             w_min=0.01,      # Min wealth
                             w_max=5.0,       # Max wealth
                             w_size=150,      # Grid size
                             ρ=0.9,           # Income persistence
                             ν=0.1,           # Income volatility
                             y_size=100):     # Income grid size
    """
    A function that takes in parameters and returns a Model instance
    for the optimal savings problem.
    """
    # Build grids and transition probabilities
    w_grid = np.linspace(w_min, w_max, w_size)
    mc = qe.tauchen(n=y_size, rho=ρ, sigma=ν)
    y_grid, Q = np.exp(mc.state_values), mc.P
    # Pack and return
    return Model(β=β, R=R, γ=γ, w_grid=w_grid, y_grid=y_grid, Q=Q)
```

(The function returns sizes of arrays because we use them later to help
compile functions in JAX.)

To produce efficient NumPy code, we will use a vectorized approach. 

The first step is to create the right hand side of the Bellman equation as a
multi-dimensional array with dimensions over all states and controls.

```{code-cell} ipython3
def B(v: np.ndarray, model: Model) -> np.ndarray:
    """
    A vectorized version of the right-hand side of the Bellman equation
    (before maximization), which is a 3D array representing

        B(w, y, w′) = u(Rw + y - w′) + β Σ_y′ v(w′, y′) Q(y, y′)

    for all (w, y, w′).
    """

    # Unpack
    β, R, γ = model.β, model.R, model.γ
    w_grid, y_grid, Q = model.w_grid, model.y_grid, model.Q
    w_size, y_size = len(w_grid), len(y_grid)

    # Compute current rewards r(w, y, wp) as array r[i, j, ip]
    w  = np.reshape(w_grid, (w_size, 1, 1))    # w[i]   ->  w[i, j, ip]
    y  = np.reshape(y_grid, (1, y_size, 1))    # z[j]   ->  z[i, j, ip]
    wp = np.reshape(w_grid, (1, 1, w_size))    # wp[ip] -> wp[i, j, ip]
    c = R * w + y - wp

    # Calculate continuation rewards at all combinations of (w, y, wp)
    v = np.reshape(v, (1, 1, w_size, y_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = np.reshape(Q, (1, y_size, 1, y_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    EV = np.sum(v * Q, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return np.where(c > 0, c**(1-γ)/(1-γ) + β * EV, -np.inf)
```

Here are two functions we need for value function iteration.

The first is the Bellman operator.

The second computes a $v$-greedy policy given $v$ (i.e., the policy that
maximizes the right-hand side of the Bellman equation.)

```{code-cell} ipython3
def T(v: np.ndarray, model: Model):
    "The Bellman operator."
    return np.max(B(v, model), axis=2)

def get_greedy(v: np.ndarray, model: Model):
    "Computes a v-greedy policy, returned as a set of indices."
    return np.argmax(B(v, model), axis=2)
```

### Value function iteration

Here's a routine that performs value function iteration.

```{code-cell} ipython3
def value_function_iteration(
        model: Model,              # Model instance
        max_iter: int = 10_000,    # max iteration bound
        tol: float = 1e-5          # error tolerance
    ):
    v = np.zeros((len(model.w_grid), len(model.y_grid)))
    i, error = 0, tol + 1
    while error > tol and i < max_iter:
        v_new = T(v, model)
        error = np.max(np.abs(v_new - v))
        i += 1
        v = v_new
    return v, get_greedy(v, model)
```

Now we create an instance, unpack it, and test how long it takes to solve the
model.

```{code-cell} ipython3
model = create_consumption_model()
# Unpack
β, R, γ = model.β, model.R, model.γ
w_size, y_size = len(model.w_grid), len(model.y_grid)
w_grid, y_grid, Q = model.w_grid, model.y_grid, model.Q
```

```{code-cell} ipython3
print("Starting NumPy VFI...")
start = time()
v_star, σ_star = value_function_iteration(model)
numpy_time = time() - start
print(f"NumPy VFI completed in {numpy_time:.3f} seconds")
```

Here's a plot of the policy function.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Policy function
    name: policy-function
---
fig, ax = plt.subplots()
ax.plot(w_grid, w_grid, "k--", label="45")
ax.plot(w_grid, w_grid[σ_star[:, 1]], label=r"$\sigma^*(\cdot, y_1)$")
ax.plot(w_grid, w_grid[σ_star[:, -1]], label=r"$\sigma^*(\cdot, y_N)$")
ax.legend()
plt.show()
```

## Switching to JAX

To switch over to JAX, we change `np` to `jnp` throughout and add some
`jax.jit` requests.


### Functions and operators

We redefine `create_consumption_model` to produce JAX arrays.

(prgm:create-consumption-model)=

```{code-cell} ipython3
def create_jax_model(
        R=1.01,          # Gross interest rate
        β=0.98,          # Discount factor
        γ=2,             # CRRA parameter
        w_min=0.01,      # Min wealth
        w_max=5.0,       # Max wealth
        w_size=150,      # Grid size
        ρ=0.9,           # Income persistence
        ν=0.1,           # Income volatility
        y_size=100       # Income grid size
    ): 
    """
    A function that takes in parameters and returns a Model instance
    with JAX arrays for the optimal savings problem.
    """
    # Create numpy model first
    numpy_model = create_consumption_model(
        R, β, γ, w_min, w_max, w_size, ρ, ν, y_size
    )

    # Convert arrays to JAX
    w_grid_jax = jnp.array(numpy_model.w_grid)
    y_grid_jax = jnp.array(numpy_model.y_grid)
    Q_jax = jnp.array(numpy_model.Q)

    return Model(β=β, R=R, γ=γ, w_grid=w_grid_jax, y_grid=y_grid_jax, Q=Q_jax)
```

The right hand side of the Bellman equation is the same as the NumPy version
after switching `np` to `jnp`.

```{code-cell} ipython3
@jax.jit
def B(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    """
    A vectorized version of the right-hand side of the Bellman equation
    (before maximization), which is a 3D array representing

        B(w, y, w′) = u(Rw + y - w′) + β Σ_y′ v(w′, y′) Q(y, y′)

    for all (w, y, w′).
    """

    # Unpack
    β, R, γ = model.β, model.R, model.γ
    w_grid, y_grid, Q = model.w_grid, model.y_grid, model.Q
    w_size, y_size = len(w_grid), len(y_grid)

    # Compute current rewards r(w, y, wp) as array r[i, j, ip]
    w  = jnp.reshape(w_grid, (w_size, 1, 1))    # w[i]   ->  w[i, j, ip]
    y  = jnp.reshape(y_grid, (1, y_size, 1))    # z[j]   ->  z[i, j, ip]
    wp = jnp.reshape(w_grid, (1, 1, w_size))    # wp[ip] -> wp[i, j, ip]
    c = R * w + y - wp

    # Calculate continuation rewards at all combinations of (w, y, wp)
    v = jnp.reshape(v, (1, 1, w_size, y_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, y_size, 1, y_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    EV = jnp.sum(v * Q, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return jnp.where(c > 0, c**(1-γ)/(1-γ) + β * EV, -jnp.inf)
```

Some readers might be concerned that we are creating high dimensional arrays,
leading to inefficiency.

Could they be avoided by more careful vectorization?

In fact this is not necessary: this function will be JIT-compiled by JAX, and
the JIT compiler will optimize compiled code to minimize memory use.

+++

In the call above, we indicate to the compiler that `sizes` is static, so the
compiler can parallelize optimally while taking array sizes as fixed.

The Bellman operator $T$ can be implemented by

```{code-cell} ipython3
@jax.jit
def T(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    "The Bellman operator."
    return jnp.max(B(v, model), axis=2)

```

The next function computes a $v$-greedy policy given $v$ (i.e., the policy that
maximizes the right-hand side of the Bellman equation.)

```{code-cell} ipython3
@jax.jit
def get_greedy(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    "Computes a v-greedy policy, returned as a set of indices."
    return jnp.argmax(B(v, model), axis=2)

```

### Successive approximation

Now we define a solver that implements VFI.

We could use the one we built for NumPy above, after changing `np` to `jnp`.

Alternatively, we can push a bit harder and write a compiled version using
`jax.lax.while_loop`.

This will give us just a bit more speed.

The first step is to write a compiled successive approximation routine that
performs fixed point iteration on some given function `T`.

```{code-cell} ipython3
@partial(jax.jit, static_argnums=(0,))
def successive_approx_jax(T,                     # Operator (callable)
                          x_0,                   # Initial condition
                          tolerance=1e-6,        # Error tolerance
                          max_iter=10_000):      # Max iteration bound
    def body_fun(k_x_err):
        k, x, error = k_x_err
        x_new = T(x)
        error = jnp.max(jnp.abs(x_new - x))
        return k + 1, x_new, error

    def cond_fun(k_x_err):
        k, x, error = k_x_err
        return jnp.logical_and(error > tolerance, k < max_iter)

    k, x, error = jax.lax.while_loop(cond_fun, body_fun,
                                    (1, x_0, tolerance + 1))
    return x
```

Our value function iteration routine calls `successive_approx_jax` while passing
in the Bellman operator.

```{code-cell} ipython3
def value_function_iteration(
        model: Model,           # Model instance
        tol: float = 1e-5       # Error tolerance
    ):
    "Perform value function iteration."
    vz = jnp.zeros((len(model.w_grid), len(model.y_grid)))
    _T = lambda v: T(v, model)
    v_star = successive_approx_jax(_T, vz, tolerance=tol)
    return v_star, get_greedy(v_star, model)
```

### Timing

Let's create an instance and unpack it.

```{code-cell} ipython3
model = create_jax_model()
# Unpack
β, R, γ = model.β, model.R, model.γ
w_size, y_size = len(model.w_grid), len(model.y_grid)
w_grid, y_grid, Q = model.w_grid, model.y_grid, model.Q
```

Let's see how long it takes to solve this model.

```{code-cell} ipython3
print("Starting JAX VFI (with compilation)...")
start = time()
v_star_jax, σ_star_jax = value_function_iteration(model)
jax_with_compile = time() - start
print(f"JAX VFI (with compile) completed in {jax_with_compile:.3f} seconds")
```

Let's run it again to eliminate compile time.

```{code-cell} ipython3
print("Running JAX VFI again (without compilation)...")
start = time()
v_star_jax, σ_star_jax = value_function_iteration(model)
jax_without_compile = time() - start
print(f"JAX VFI (without compile) completed in {jax_without_compile:.3f} seconds")
```

The relative speed gain is

```{code-cell} ipython3
speedup = numpy_time / jax_without_compile
print(f"\n--- Performance Comparison ---")
print(f"NumPy time:     {numpy_time:.3f} seconds")
print(f"JAX time:       {jax_without_compile:.3f} seconds")
print(f"Speedup:        {speedup:.1f}x faster")
```

This is an impressive speed up and in fact we can do better still by switching
to alternative algorithms that are better suited to parallelization.

These algorithms are discussed in a {doc}`separate lecture <opt_savings_2>`.


## Switching to vmap

Before we discuss alternative algorithms, let's take another look at value
function iteration.

For this simple optimal savings problem, direct vectorization is relatively easy.

In particular, it's straightforward to express the right hand side of the
Bellman equation as an array that stores evaluations of the function at every
state and control.

For more complex models direct vectorization can be much harder.

For this reason, it helps to have another approach to fast JAX implementations
up our sleeves.

Here's a version that 

1. writes the right hand side of the Bellman operator as a function of individual states and controls, and 
1. applies `jax.vmap` on the outside to achieve a parallelized solution.

First let's rewrite `B`

```{code-cell} ipython3
def B(
        v: jnp.ndarray,   # guess of value function
        model: Model,     # instance of jax Model
        i: int,           # current wealth index
        j: int,           # current income index
        ip: int           # future wealth index
    ):
    """
    The right-hand side of the Bellman equation before maximization, which takes
    the form

        B(w, y, w′) = u(Rw + y - w′) + β Σ_y′ v(w′, y′) Q(y, y′)

    The indices are (i, j, ip) -> (w, y, w′).
    """
    β, R, γ = model.β, model.R, model.γ
    w_grid, y_grid, Q = model.w_grid, model.y_grid, model.Q
    w, y, wp  = w_grid[i], y_grid[j], w_grid[ip]
    c = R * w + y - wp
    EV = jnp.sum(v[ip, :] * Q[j, :])
    return jnp.where(c > 0, c**(1-γ)/(1-γ) + β * EV, -jnp.inf)
```

Now we successively apply `vmap` to simulate nested loops.

```{code-cell} ipython3
B_1    = jax.vmap(B,   in_axes=(None, None, None, None, 0))
B_2    = jax.vmap(B_1, in_axes=(None, None, None, 0,    None))
B_vmap = jax.vmap(B_2, in_axes=(None, None, 0,    None, None))
```

Here's the Bellman operator and the `get_greedy` functions for the `vmap` case.

```{code-cell} ipython3
@jax.jit
def T_vmap(v: jnp.ndarray, model: Model):
    "The Bellman operator implemented with vmap."

    w_size, y_size = len(model.w_grid), len(model.y_grid)
    w_indices, y_indices = jnp.arange(w_size), jnp.arange(y_size)
    B_values = B_vmap(v, model, w_indices, y_indices, w_indices)

    return jnp.max(B_values, axis=-1)



@jax.jit
def get_greedy_vmap(v: jnp.ndarray, model: Model):
    "Computes a v-greedy policy, returned as a set of indices."

    w_size, y_size = len(model.w_grid), len(model.y_grid)
    w_indices, y_indices = jnp.arange(w_size), jnp.arange(y_size)
    B_values = B_vmap(v, model, w_indices, y_indices, w_indices)

    return jnp.argmax(B_values, axis=-1)

```

Here's the iteration routine.

```{code-cell} ipython3
def value_iteration_vmap(
        model: Model,           # Model instance
        tol: float = 1e-5       # Error tolerance
    ):
    vz = jnp.zeros((len(model.w_grid), len(model.y_grid)))
    _T = lambda v: T_vmap(v, model)
    v_star = successive_approx_jax(_T, vz, tolerance=tol)
    return v_star, get_greedy(v_star, model)
```

Let's see how long it takes to solve the model using the `vmap` method.

```{code-cell} ipython3
print("\nStarting JAX vmap VFI (with compilation)...")
start = time()
v_star_vmap, σ_star_vmap = value_iteration_vmap(model)
jax_vmap_with_compile = time() - start
print(f"JAX vmap VFI (with compile) completed in {jax_vmap_with_compile:.3f} seconds")
```

Let's run it again to get rid of compile time.

```{code-cell} ipython3
print("Running JAX vmap VFI again (without compilation)...")
start = time()
v_star_vmap, σ_star_vmap = value_iteration_vmap(model)
jax_vmap_without_compile = time() - start
print(f"JAX vmap VFI (without compile) completed in {jax_vmap_without_compile:.3f} seconds")
```

We need to make sure that we got the same result.

```{code-cell} ipython3
value_match = jnp.allclose(v_star_vmap, v_star_jax)
policy_match = jnp.allclose(σ_star_vmap, σ_star_jax)
print(f"\n--- Verification ---")
print(f"Value functions match:  {value_match}")
print(f"Policy functions match: {policy_match}")
```

Here's the speed gain associated with switching from the NumPy version to JAX with `vmap`:

```{code-cell} ipython3
vmap_speedup = numpy_time / jax_vmap_without_compile
print(f"\n--- NumPy vs JAX vmap Comparison ---")
print(f"NumPy time:     {numpy_time:.3f} seconds")
print(f"JAX vmap time:  {jax_vmap_without_compile:.3f} seconds")
print(f"Speedup:        {vmap_speedup:.1f}x faster")
```

And here's the comparison with the first JAX implementation (which used direct vectorization).

```{code-cell} ipython3
vmap_vs_vectorized = jax_without_compile / jax_vmap_without_compile
print(f"\n--- JAX Method Comparison ---")
print(f"JAX vectorized: {jax_without_compile:.3f} seconds")
print(f"JAX vmap:       {jax_vmap_without_compile:.3f} seconds")
print(f"Ratio:          {vmap_vs_vectorized:.2f}x (vectorized vs vmap)")
```


The execution times for the two JAX versions are relatively similar.

However, as emphasized above, having a second method up our sleeves (i.e, the
`vmap` approach) will be helpful when confronting dynamic programs with more
sophisticated Bellman equations.
