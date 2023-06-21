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

```{code-cell} ipython3
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from collections import namedtuple
```

```{code-cell} ipython3
@jax.jit
def c_star(x, β, γ):
    return (1 - β ** (1/γ)) * x

@jax.jit
def v_star(x, β, γ):
    return (1 - β**(1 / γ))**(-γ) * (x**(1-γ) / (1-γ))
```

```{code-cell} ipython3
CEM = namedtuple('CakeEatingModel',
                    ('β', 'γ', 'x_grid'))
```

```{code-cell} ipython3
def create_cake_eating_model(β=0.96,           # discount factor
                             γ=1.5,            # degree of relative risk aversion
                             x_grid_min=1e-3,  # exclude zero for numerical stability
                             x_grid_max=2.5,   # size of cake
                             x_grid_size=120):
    x_grid = jnp.linspace(x_grid_min, x_grid_max, x_grid_size)
    return CEM(β=β, γ=γ, x_grid=x_grid)
```

```{code-cell} ipython3
# Utility function
@jax.jit
def u(c, cem):
    return jnp.where(cem.γ == 1, jnp.log(c), (c ** (1 - cem.γ)) / (1 - cem.γ))
```

```{code-cell} ipython3
@jax.jit
def state_action_value(x, c, v_array, cem):
    """
    Right hand side of the Bellman equation given x and c.
    """

    return jnp.where(c <= x,
                     u(c, cem) + cem.β * jnp.interp(x - c, cem.x_grid, v_array),
                     -jnp.inf)
```

```{code-cell} ipython3
state_action_value_vec = jax.vmap(state_action_value, (0, None, None, None))
```

```{code-cell} ipython3
@jax.jit
def maximize(x, v_array, cem):
    """
    Maximize the function g over the interval (0, x).

    We use the fact that the maximizer of g on any interval is
    also the minimizer of -g.  The tuple args collects any extra
    arguments to g.

    Returns the maximal value and the maximizer.
    """
    c_grid = jnp.linspace(1e-10, x.max(), 100_000)
    return jnp.max(state_action_value_vec(x, c_grid, v_array, cem), axis=1)
```

```{code-cell} ipython3
@jax.jit
def T(v, ce):
    """
    The Bellman operator.  Updates the guess of the value function.

    * ce is an instance of CakeEating
    * v is an array representing a guess of the value function

    """
    return maximize(ce.x_grid, v, ce)
```

```{code-cell} ipython3
ce = create_cake_eating_model()
```

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
v = compute_value_function(ce)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_grid, v, label='Approximate value function')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.set_title('Value function')
ax.legend()
plt.show()
```

```{code-cell} ipython3
v_analytical = v_star(ce.x_grid, ce.β, ce.γ)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(x_grid, v_analytical, label='analytical solution')
ax.plot(x_grid, v, label='numerical solution')
ax.set_ylabel('$V(x)$', fontsize=12)
ax.set_xlabel('$x$', fontsize=12)
ax.legend()
ax.set_title('Comparison between analytical and numerical value functions')
plt.show()
```

```{code-cell} ipython3
@jax.jit
def maximizer(x, v_array, cem):
    """
    Returns the maximizer.
    """
    c_grid = jnp.linspace(1e-10, x.max(), 100_000)
    i_cs =  jnp.argmax(state_action_value_vec(x, c_grid, v_array, cem), axis=1)
    return c_grid[i_cs]
```

```{code-cell} ipython3
@jax.jit
def σ(ce, v):
    """
    The optimal policy function. Given the value function,
    it finds optimal consumption in each state.

    * ce is an instance of CakeEating
    * v is a value function array

    """
    return maximizer(ce.x_grid, v, ce)
```

```{code-cell} ipython3
c = σ(ce, v)
```

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
