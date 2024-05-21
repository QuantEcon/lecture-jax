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

# Cake Eating: Numerical Methods

```{include} _admonition/gpu.md
```

This lecture is the extended JAX implementation of [this lecture](https://python.quantecon.org/cake_eating_numerical.html).

Please refer that lecture for all background and notation.

In addition to JAX and Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

We will use the following imports.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import namedtuple
import time
```

Let's check the GPU we are running

```{code-cell} ipython3
!nvidia-smi
```

## Reviewing the Model

Recall in particular that the Bellman equation is

```{math}
:label: bellman-cen

v(x) = \max_{0\leq c \leq x} \{u(c) + \beta v(x-c)\}
\quad \text{for all } x \geq 0.
```

where $u$ is the CRRA utility function.


## Implementation using JAX


The analytical solutions for the value function and optimal policy were found
to be as follows.

```{code-cell} ipython3
@jax.jit
def c_star(x, β, γ):
    return (1 - β ** (1/γ)) * x

@jax.jit
def v_star(x, β, γ):
    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))
```

Let's define a model to represent the Cake Eating Problem.

```{code-cell} ipython3
CEM = namedtuple('CakeEatingModel',
                    ('β', 'γ', 'x_grid', 'c_grid'))
```

```{code-cell} ipython3
def create_cake_eating_model(β=0.96,           # discount factor
                             γ=1.5,            # degree of relative risk aversion
                             x_grid_min=1e-3,  # exclude zero for numerical stability
                             x_grid_max=2.5,   # size of cake
                             x_grid_size=200):
    x_grid = jnp.linspace(x_grid_min, x_grid_max, x_grid_size)

    # c_grid used for finding maximize function values using brute force
    c_grid = jnp.linspace(x_grid_min, x_grid_max, 100*x_grid_size)
    return CEM(β=β, γ=γ, x_grid=x_grid, c_grid=c_grid)
```

Now let's define the CRRA utility function.

```{code-cell} ipython3
# Utility function
@jax.jit
def u(c, cem):
    return (c ** (1 - cem.γ)) / (1 - cem.γ)
```

### The Bellman Operator

We introduce the **Bellman operator** $T$ that takes a function v as an
argument and returns a new function $Tv$ defined by

$$
Tv(x) = \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}
$$

From $v$ we get $Tv$, and applying $T$ to this yields
$T^2 v := T (Tv)$ and so on.

This is called **iterating with the Bellman operator** from initial guess
$v$.

```{code-cell} ipython3
@jax.jit
def state_action_value(x, c, v_array, ce):
    """
    Right hand side of the Bellman equation given x and c.
    * x: scalar element `x`
    * c: c_grid, 1-D array
    * v_array: value function array guess, 1-D array
    * ce: Cake Eating Model instance
    """

    return jnp.where(c <= x,
                     u(c, ce) + ce.β * jnp.interp(x - c, ce.x_grid, v_array),
                     -jnp.inf)
```

In order to create a vectorized function using `state_action_value`, we use [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html).
This function returns a new vectorized version of the above function which is vectorized on the argument `x`.

```{code-cell} ipython3
state_action_value_vec = jax.vmap(state_action_value, (0, None, None, None))
```

```{code-cell} ipython3
@jax.jit
def T(v, ce):
    """
    The Bellman operator. Updates the guess of the value function.

    * ce: Cake Eating Model instance
    * v: value function array guess, 1-D array

    """
    return jnp.max(state_action_value_vec(ce.x_grid, ce.c_grid, v, ce), axis=1)
```

Let’s start by creating a Cake Eating Model instance using the default parameterization.

```{code-cell} ipython3
ce = create_cake_eating_model()
```

Now let's see the iteration of the value function in action.

We start from guess $v$ given by $v(x) = u(x)$ for every
$x$ grid point.

```{code-cell} ipython3
x_grid = ce.x_grid
v = u(x_grid, ce)       # Initial guess
n = 12                 # Number of iterations

fig, ax = plt.subplots()

ax.plot(x_grid, v, color=plt.cm.jet(0),
        lw=2, alpha=0.6, label='Initial guess')

for i in range(n):
    v = T(v, ce)  # Apply the Bellman operator
    ax.plot(x_grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

ax.legend()
ax.set_ylabel('value', fontsize=12)
ax.set_xlabel('cake size $x$', fontsize=12)
ax.set_title('Value function iterations')

plt.show()
```

Let's introduce a wrapper function called `compute_value_function`
that iterates until some convergence conditions are satisfied.

```{code-cell} ipython3
def compute_value_function(ce,
                           tol=1e-4,
                           max_iter=1000,
                           verbose=True,
                           print_skip=25):

    # Set up loop
    v = jnp.zeros(len(ce.x_grid)) # Initial guess
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T(v, ce)

        error = jnp.max(jnp.abs(v - v_new))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")

        v = v_new

    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return v_new
```

```{code-cell} ipython3
in_time = time.time()
v_jax = compute_value_function(ce)
jax_time = time.time() - in_time
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_grid, v_jax, label='Approximate value function')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.set_title('Value function')
ax.legend()
plt.show()
```

Next let’s compare it to the analytical solution.

```{code-cell} ipython3
v_analytical = v_star(ce.x_grid, ce.β, ce.γ)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_grid, v_analytical, label='analytical solution')
ax.plot(x_grid, v_jax, label='numerical solution')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.legend()
ax.set_title('Comparison between analytical and numerical value functions')
plt.show()
```

### Policy Function

Recall that the optimal consumption policy was shown to be

$$
\sigma^*(x) = \left(1-\beta^{1/\gamma} \right) x
$$

Let's see if our numerical results lead to something similar.

Our numerical strategy will be to compute

$$
\sigma(x) = \arg \max_{0 \leq c \leq x} \{u(c) + \beta v(x - c)\}
$$

on a grid of $x$ points and then interpolate.

For $v$ we will use the approximation of the value function we obtained
above.

Here's the function:

```{code-cell} ipython3
@jax.jit
def σ(ce, v):
    """
    The optimal policy function. Given the value function,
    it finds optimal consumption in each state.

    * ce: Cake Eating Model instance
    * v: value function array guess, 1-D array

    """
    i_cs =  jnp.argmax(state_action_value_vec(ce.x_grid, ce.c_grid, v, ce), axis=1)
    return ce.c_grid[i_cs]
```

Now let’s pass the approximate value function and compute optimal consumption:

```{code-cell} ipython3
c = σ(ce, v_jax)
```

Let’s plot this next to the true analytical solution

```{code-cell} ipython3
c_analytical = c_star(ce.x_grid, ce.β, ce.γ)

fig, ax = plt.subplots()

ax.plot(ce.x_grid, c_analytical, label='analytical')
ax.plot(ce.x_grid, c, label='numerical')
ax.set_ylabel(r'$\sigma(x)$')
ax.set_xlabel('$x$')
ax.legend()

plt.show()
```

## Numba implementation

This section of the lecture is directly adapted from [this lecture](https://python.quantecon.org/cake_eating_numerical.html)
for the purpose of comparing the results of JAX implementation.

```{code-cell} ipython3
import numpy as np
from numba import prange, jit
from quantecon.optimize import brent_max
```

```{code-cell} ipython3
CEMN = namedtuple('CakeEatingModelNumba',
                    ('β', 'γ', 'x_grid'))
```

```{code-cell} ipython3
def create_cake_eating_model_numba(β=0.96,           # discount factor
                                   γ=1.5,            # degree of relative risk aversion
                                   x_grid_min=1e-3,  # exclude zero for numerical stability
                                   x_grid_max=2.5,   # size of cake
                                   x_grid_size=200):
    x_grid = np.linspace(x_grid_min, x_grid_max, x_grid_size)
    return CEMN(β=β, γ=γ, x_grid=x_grid)
```

```{code-cell} ipython3
# Utility function
@jit
def u_numba(c, cem):
    return (c ** (1 - cem.γ)) / (1 - cem.γ)
```

```{code-cell} ipython3
@jit
def state_action_value_numba(c, x, v_array, cem):
    """
    Right hand side of the Bellman equation given x and c.
    * x: scalar element `x`
    * c: consumption
    * v_array: value function array guess, 1-D array
    * cem: Cake Eating Numba Model instance
    """
    return u_numba(c, cem) + cem.β * np.interp(x - c, cem.x_grid, v_array)
```

```{code-cell} ipython3
@jit
def T_numba(v, ce):
    """
    The Bellman operator.  Updates the guess of the value function.

    * ce is an instance of CakeEatingNumba Model
    * v is an array representing a guess of the value function

    """
    v_new = np.empty_like(v)

    for i in prange(len(ce.x_grid)):
        # Maximize RHS of Bellman equation at state x
        v_new[i] = brent_max(state_action_value_numba, 1e-10, ce.x_grid[i],
                             args=(ce.x_grid[i], v, ce))[1]
    return v_new
```

```{code-cell} ipython3
def compute_value_function_numba(ce,
                           tol=1e-4,
                           max_iter=1000,
                           verbose=True,
                           print_skip=25):

    # Set up loop
    v = np.zeros(len(ce.x_grid)) # Initial guess
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T_numba(v, ce)

        error = np.max(np.abs(v - v_new))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")

        v = v_new

    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return v_new
```

```{code-cell} ipython3
cen = create_cake_eating_model_numba()
```

```{code-cell} ipython3
in_time = time.time()
v_np = compute_value_function_numba(cen)
numba_time = time.time() - in_time
```

```{code-cell} ipython3
ratio = numba_time/jax_time
print(f"JAX implementation is {ratio} times faster than Numba.")
print(f"JAX time: {jax_time}")
print(f"Numba time: {numba_time}")
```
