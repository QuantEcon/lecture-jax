---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Inventory Dynamics

```{include} _admonition/gpu.md
```

```{index} single: Markov process, inventory
```

## Overview

This lecture explores the inventory dynamics of a firm using so-called s-S inventory control.

Loosely speaking, this means that the firm 

* waits until inventory falls below some value $s$
* and then restocks with a bulk order of $S$ units (or, in some models, restocks up to level $S$).

We will be interested in the distribution of the associated Markov process,
which can be thought of as cross-sectional distributions of inventory levels
across a large number of firms, all of which 

1. evolve independently and
1. have the same dynamics.

Note that we also studied this model in a [separate
lecture](https://python.quantecon.org/inventory_dynamics.html), using Numba.

Here we study the same problem using JAX.

We will use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from collections import namedtuple
```

Here's a description of our GPU:

```{code-cell} ipython3
!nvidia-smi
```

## Sample paths

Consider a firm with inventory $X_t$.

The firm waits until $X_t \leq s$ and then restocks up to $S$ units.

It faces stochastic demand $\{ D_t \}$, which we assume is IID across time and
firms.

With notation $a^+ := \max\{a, 0\}$, inventory dynamics can be written
as

$$
X_{t+1} =
    \begin{cases}
      ( S - D_{t+1})^+ & \quad \text{if } X_t \leq s \\
      ( X_t - D_{t+1} )^+ &  \quad \text{if } X_t > s
    \end{cases}
$$

(See our earlier [lecture on inventory dynamics](https://python.quantecon.org/inventory_dynamics.html) for background and motivation.)

In what follows, we will assume that each $D_t$ is lognormal, so that

$$
    D_t = \exp(\mu + \sigma Z_t)
$$

where $\mu$ and $\sigma$ are parameters and $\{Z_t\}$ is IID
and standard normal.

Here's a `namedtuple` that stores parameters.

```{code-cell} ipython3
Parameters = namedtuple('Parameters', ['s', 'S', 'mu', 'sigma'])

# Create a default instance
params = Parameters(s=10, S=100, mu=1.0, sigma=0.5)
```

## Marginal distributions

Now let’s look at the marginal distribution $\psi_T$ of $X_T$ for some fixed $T$.

We will approximate this distribution by 

1. fixing $n$ to be some large number, indicating the number of firms in the
   simulation,
1. fixing $T$, the time period we are interested in,
1. generating $n$ independent draws from some fixed distribution $\psi_0$ that gives the
   initial cross-sectional distribution of firms, and
1. shift this distribution forward in time $T$ periods, by updating each firm
   independently via the dynamics described above.

This process gives us $\psi_T$, a distribution of firm inventory levels.

We will then use various methods to visualize $\psi_T$, such as historgrams and
kernel density estimates.

We will use the following code to update the cross-section of firms by one period.

```{code-cell} ipython3
@jax.jit
def update_cross_section(params, X_vec, D):
    """
    Update by one period a cross-section of firms with inventory levels given by
    X_vec, given the vector of demand shocks in D.

       * D[i] is the demand shock for firm i with current inventory X_vec[i]

    """
    # Unpack
    s, S = params.s, params.S
    # Restock if the inventory is below the threshold
    X_new = jnp.where(X_vec <= s, 
                      jnp.maximum(S - D, 0), jnp.maximum(X_vec - D, 0))
    return X_new
```

### For loop version

Here's code to compute the cross-sectional distribution $\psi_T$ given some
initial distribution $\psi_0$ and a positive integer $T$.

In this code we use an ordinary Python `for` loop, which reasonable here because
efficiency of outer loops has less influence on runtime than efficiency of inner loops.

(Below we will squeeze out more speed by compiling the outer loop as well as the
update rule.)

In the code below, the initial distribution $\psi_0$ takes all firms to have
initial inventory `x_init`.

```{code-cell} ipython3
def compute_cross_section(params, x_init, T, key, num_firms=50_000):
    # Set up initial distribution
    X_vec = jnp.full((num_firms, ), x_init)
    # Loop
    for i in range(T):
        Z = random.normal(key, shape=(num_firms, ))
        D = jnp.exp(params.mu + params.sigma * Z)

        X_vec = update_cross_section(params, X_vec, D)
        _, key = random.split(key)

    return X_vec
```

We'll use the following specification

```{code-cell} ipython3
x_init = 50
T = 500
# Initialize random number generator
key = random.PRNGKey(10)
```

Let's look at the timing.

```{code-cell} ipython3
%time X_vec = compute_cross_section(params, \
        x_init, T, key).block_until_ready()
```

```{code-cell} ipython3
%time X_vec = compute_cross_section(params, \
        x_init, T, key).block_until_ready()
```

Here's a histogram of inventory levels at time $T$.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(X_vec, bins=50, 
        density=True, 
        histtype='step', 
        label=f'cross-section when $t = {T}$')
ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```




### Compiling the outer loop

Now let's see if we can gain some speed by compiling the outer loop, which steps
through the time dimension.

We will do this using `jax.jit` and a `fori_loop`, which is a compiler-ready version of a for loop provided by JAX.



```{code-cell} ipython3
def compute_cross_section_fori(params, x_init, T, key, num_firms=50_000):

    s, S, mu, sigma = params.s, params.S, params.mu, params.sigma
    X = jnp.full((num_firms, ), x_init)

    # Define the function for each update
    def update_cross_section(i, inputs):
        X, key = inputs
        Z = random.normal(key, shape=(num_firms,))
        D = jnp.exp(mu + sigma * Z)
        X = jnp.where(X <= s,
                  jnp.maximum(S - D, 0),
                  jnp.maximum(X - D, 0))
        key, subkey = random.split(key)
        return X, subkey

    # Use lax.scan to perform the calculations on all states
    X, key = lax.fori_loop(0, T, update_cross_section, (X, key))

    return X

# Compile taking T and num_firms as static (changes trigger recompile)
compute_cross_section_fori = jax.jit(
    compute_cross_section_fori, static_argnums=(2, 4))
```

Let's see how fast this runs with compile time.

```{code-cell} ipython3
%time X_vec = compute_cross_section_fori(params, \
    x_init, T, key).block_until_ready()
```

And let's see how fast it runs without compile time.

```{code-cell} ipython3
%time X_vec = compute_cross_section_fori(params, \
    x_init, T, key).block_until_ready()
```

Compared to the original version with a pure Python outer loop, we have 
produced a nontrivial speed gain.


This is due to the fact that we have compiled the whole operation.




### Further vectorization

For relatively small problems, we can make this code run even faster by generating
all random variables at ones.

This improves efficiency because we are taking more operations out of the loop.

```{code-cell} ipython3
def compute_cross_section_fori(params, x_init, T, key, num_firms=50_000):

    s, S, mu, sigma = params.s, params.S, params.mu, params.sigma
    X = jnp.full((num_firms, ), x_init)
    Z = random.normal(key, shape=(T, num_firms))
    D = jnp.exp(mu + sigma * Z)

    # Define the function for each update
    def update_cross_section(i, X):
        X = jnp.where(X <= s,
                  jnp.maximum(S - D[i, :], 0),
                  jnp.maximum(X - D[i, :], 0))
        return X

    # Use lax.scan to perform the calculations on all states
    X = lax.fori_loop(0, T, update_cross_section, X)

    return X

# Compile taking T and num_firms as static (changes trigger recompile)
compute_cross_section_fori = jax.jit(
    compute_cross_section_fori, static_argnums=(2, 4))
```

Let's test it with compile time included.

```{code-cell} ipython3
%time X_vec = compute_cross_section_fori(params, \
    x_init, T, key).block_until_ready()
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
%time X_vec = compute_cross_section_fori(params, \
    x_init, T, key).block_until_ready()
```

On one hand, this version is faster than the previous one, where random variables were
generated inside the loop.

On the other hand, this implementation consumes far more memory, as we need to
store large arrays of random draws.

The high memory consumption becomes problematic for large problems.



## Distribution dynamics

Let's take a look at how the distribution sequence evolves over time.

Here is code that repeatedly shifts the cross-section forward while
recording the cross-section at the dates in `sample_dates`.

```{code-cell} ipython3
def shift_forward_and_sample(x_init, params, sample_dates,
                        key, num_firms=50_000, sim_length=750):

    X = res = jnp.full((num_firms, ), x_init)

    # Use for loop to update X and collect samples
    for i in range(sim_length):
        Z = random.normal(key, shape=(num_firms, ))
        D = jnp.exp(params.mu + params.sigma * Z)

        X = update_cross_section(params, X, D)
        _, key = random.split(key)

        # draw a sample at the sample dates
        if (i+1 in sample_dates):
          res = jnp.vstack((res, X))

    return res[1:]
```

Let's test it

```{code-cell} ipython3
x_init = 50
num_firms = 10_000
sample_dates = 10, 50, 250, 500, 750
key = random.PRNGKey(10)


%time X = shift_forward_and_sample(x_init, params, \
                              sample_dates, key).block_until_ready()
```

Let's plot the output.

```{code-cell} ipython3
fig, ax = plt.subplots()

for i, date in enumerate(sample_dates):
    ax.hist(X[i, :], bins=50, 
            density=True, 
            histtype='step',
            label=f'cross-section when $t = {date}$')

ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

This model for inventory dynamics is asymptotically stationary, with a unique
stationary distribution.

In particular, the sequence of marginal distributions $\{\psi_t\}$
converges to a unique limiting distribution that does not depend on
initial conditions.

Although we will not prove this here, we can see it in the simulation above.

By $t=500$ or $t=750$ the distributions are barely changing.

If you test a few different initial conditions, you will see that they do not affect long-run outcomes.





## Restock frequency

As an exercise, let's study the probability that firms need to restock over a given time perion.

In the exercise, we will

* set the starting stock level to $X_0 = 70$ and
* calculate the proportion of firms that need to order twice or more in the first 50 periods.

This proportion approximates the probability of the event when the sample size
is large.


### For loop version

We start with an easier `for` loop implementation

```{code-cell} ipython3
# Define a jitted function for each update
@jax.jit
def update_stock(n_restock, X, params, D):
    n_restock = jnp.where(X <= params.s,
                          n_restock + 1,
                          n_restock)
    X = jnp.where(X <= params.s,
                  jnp.maximum(params.S - D, 0),
                  jnp.maximum(X - D, 0))
    return n_restock, X, key

def compute_freq(params, key,
                 x_init=70,
                 sim_length=50,
                 num_firms=1_000_000):

    # Prepare initial arrays
    X = jnp.full((num_firms, ), x_init)

    # Stack the restock counter on top of the inventory
    n_restock = jnp.zeros((num_firms, ))

    # Use a for loop to perform the calculations on all states
    for i in range(sim_length):
        Z = random.normal(key, shape=(num_firms, ))
        D = jnp.exp(params.mu + params.sigma * Z)
        n_restock, X, key = update_stock(
            n_restock, X, params, D)
        key = random.fold_in(key, i)

    return jnp.mean(n_restock > 1, axis=0)
```

```{code-cell} ipython3
key = random.PRNGKey(27)
%time freq = compute_freq(params, key).block_until_ready()
print(f"Frequency of at least two stock outs = {freq}")
```

### Alternative implementation with `lax.fori_loop`

Now let's write a `lax.fori_loop` version that JIT compiles the whole function

```{code-cell} ipython3
@jax.jit
def compute_freq(params, key,
                 x_init=70,
                 sim_length=50,
                 num_firms=1_000_000):

    s, S, mu, sigma = params.s, params.S, params.mu, params.sigma
    # Prepare initial arrays
    X = jnp.full((num_firms, ), x_init)
    Z = random.normal(key, shape=(sim_length, num_firms))
    D = jnp.exp(mu + sigma * Z)

    # Stack the restock counter on top of the inventory
    restock_count = jnp.zeros((num_firms, ))
    Xs = (X, restock_count)

    # Define the function for each update
    def update_cross_section(i, Xs):
        # Separate the inventory and restock counter
        x, restock_count = Xs[0], Xs[1]
        restock_count = jnp.where(x <= s,
                                restock_count + 1,
                                restock_count)
        x = jnp.where(x <= s,
                      jnp.maximum(S - D[i], 0),
                      jnp.maximum(x - D[i], 0))

        Xs = (x, restock_count)
        return Xs

    # Use lax.fori_loop to perform the calculations on all states
    X_final = lax.fori_loop(0, sim_length, update_cross_section, Xs)

    return jnp.mean(X_final[1] > 1)
```

Note the time the routine takes to run, as well as the output

```{code-cell} ipython3
%time freq = compute_freq(params, key).block_until_ready()
%time freq = compute_freq(params, key).block_until_ready()

print(f"Frequency of at least two stock outs = {freq}")
```

```{code-cell} ipython3

```
