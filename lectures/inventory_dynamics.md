---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
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

```{index} single: Markov process, inventory
```

```{contents} Contents
:depth: 2
```

We will use the following imports:

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from collections import namedtuple
```

## Sample Paths

Consider a firm with inventory $X_t$.

The firm waits until $X_t \leq s$ and then restocks up to $S$ units.

It faces stochastic demand $\{ D_t \}$, which we assume is IID.

With notation $a^+ := \max\{a, 0\}$, inventory dynamics can be written
as

$$
X_{t+1} =
    \begin{cases}
      ( S - D_{t+1})^+ & \quad \text{if } X_t \leq s \\
      ( X_t - D_{t+1} )^+ &  \quad \text{if } X_t > s
    \end{cases}
$$

In what follows, we will assume that each $D_t$ is lognormal, so that

$$
D_t = \exp(\mu + \sigma Z_t)
$$

where $\mu$ and $\sigma$ are parameters and $\{Z_t\}$ is IID
and standard normal.

Here's a `namedtuple` that stores parameters and generates time paths for inventory.

```{code-cell} python3
Firm = namedtuple('Firm', ['s', 'S', 'mu', 'sigma'])

firm = Firm(s=10, S=100, mu=1.0, sigma=0.5)
```

## Example 1: Marginal Distributions

Now let’s look at the marginal distribution $\psi_T$ of $X_T$ for some
fixed $T$.

We can approximate the distribution using a [kernel density estimator](https://en.wikipedia.org/wiki/Kernel_density_estimation).

Kernel density estimators can be thought of as smoothed histograms.

We will use a kernel density estimator from [scikit-learn](https://scikit-learn.org/stable/)

```{code-cell} ipython3
from sklearn.neighbors import KernelDensity

def plot_kde(sample, ax, label=''):
    xmin, xmax = 0.9 * min(sample), 1.1 * max(sample)
    xgrid = np.linspace(xmin, xmax, 200)
    kde = KernelDensity(kernel='gaussian').fit(sample[:, None])
    log_dens = kde.score_samples(xgrid[:, None])

    ax.plot(xgrid, np.exp(log_dens), label=label)
```

This model for inventory dynamics is asymptotically stationary, with a unique stationary distribution.

In particular, the sequence of marginal distributions $\{\psi_t\}$
is converging to a unique limiting distribution that does not depend on
initial conditions.

Although we will not prove this here, we can investigate it using simulation.

We can generate and plot the sequence $\{\psi_t\}$ at times
$t = 10, 50, 250, 500, 750$ based on the kernel density estimator.

We will see convergence, in the sense that differences between successive distributions are getting smaller.

Here is one realization of the process in JAX using `for` loop

```{code-cell} ipython3
def shift_firms_forward(x_init, key, sample_dates, num_firms=50_000, sim_length=750):
    X = res = jnp.full((num_firms, ), x_init)

    # Define a jit-compiled function to update X and key
    @jax.jit
    def update_X(X, key):
        Z = random.normal(key, shape=(num_firms, ))
        D = jnp.exp(mu + sigma * Z)

        # Restock if the inventory is below the threshold
        X = jnp.where(X <= s,
                jnp.maximum(S - D, 0),
                jnp.maximum(X - D, 0))

        _, key = random.split(key)
        return X, key

    # Use a for loop to update X and collect samples
    for i in range(sim_length):
        X, key = update_X(X, key)
        if (i in sample_dates):
          res = jnp.vstack((res, X))

    return res[1:]
```

```{code-cell}ipython3
x_init = 50
num_firms = 50_000
sample_dates = 10, 50, 250, 500, 750
s, S, mu, sigma = firm.s, firm.S, firm.mu, firm.sigma
key = random.PRNGKey(10)

fig, ax = plt.subplots()

%time X = shift_firms_forward(x_init, key, sample_dates)

for i, date in enumerate(sample_dates):
   plot_kde(X[i, :], ax, label=f't = {date}')

ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

Note that we only compiled the function within the `for` loop as `jit` compilation of the `for` loop takes a very long time when the workload is heavy inside the `for` loop.

Since the function itself is not JIT-compiled, it also takes longer to run when we call it again.

This is why [JAX documentation](https://jax.readthedocs.io/en/latest/faq.html#jit-decorated-function-is-very-slow-to-compile) for JAX recommends `lax.scan` to perform heavy computations within a loop.

Here is an example of the same function in `lax.scan`

```{code-cell} ipython3
@jax.jit
def shift_firms_forward(x_init, key, num_firms=50_000, sim_length=750):
    X = jnp.full((num_firms, ), x_init)
    Z = random.normal(key, shape=(sim_length, num_firms))
    D = jnp.exp(mu + sigma * Z)
    
    # Define the function for each update
    def update_X(X, D):
        res = jnp.where(X <= s, 
                  jnp.maximum(S - D, 0), 
                  jnp.maximum(X - D, 0))
        return res, res

    # Use lax.scan to perform the calculations on all states
    _, X_final = lax.scan(update_X, X, D)

    return X_final
```

It is clear that `lax.scan` has a more complicated syntax and can be memory intensive as we need to have large samples of `Z` in memory.

```{code-cell} ipython3
fig, ax = plt.subplots()

%time X = shift_firms_forward(x_init, key)

for date in sample_dates:
   plot_kde(X[date, :], ax, label=f't = {date}')

ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

Notice that by $t=500$ or $t=750$ the densities are barely
changing.

We have reached a reasonable approximation of the stationary density.

You can test a few more initial conditions to show that they don’t matter by
testing.

For example, try rerunning the code above with all firms starting at
$X_0 = 20$

```{code-cell} ipython3
x_init = 20.0

fig, ax = plt.subplots()

%time X = shift_firms_forward(x_init, key)

for date in sample_dates:
   plot_kde(X[date, :], ax, label=f't = {date}')

ax.set_xlabel('inventory')
ax.set_ylabel('probability')
ax.legend()
plt.show()
```

We noticed that the compiled function runs very fast. 

## Example 2: Restock Frequency

Let's go through another example where we calculate the probability of firms having restocks.  

Specifically we set the starting stock level to 70 ($X_0 = 70$), as we calculate the proportion of firms that need to order twice or more in the first 50 periods.

You will need a large sample size to get an accurate reading.

Again, we start with an easier but slightly slower `for` loop implementation

```{code-cell} ipython3
def compute_freq(key, x_init=70, sim_length=50, num_firms=1_000_000):
    # Prepare initial arrays
    X = jnp.full((num_firms, ), x_init)

    # Stack the restock counter on top of the inventory
    restock_counter = jnp.zeros((num_firms, ))
    
    # Define a jitted function for each update
    @jax.jit
    def update_stock(restock_counter, X, key):
      Z = random.normal(key, shape=(num_firms, ))
      D = jnp.exp(mu + sigma * Z)
      restock_counter = jnp.where(X <= s,
                                restock_counter + 1,
                                restock_counter)
      X = jnp.where(X <= s,
                    jnp.maximum(S - D, 0),
                    jnp.maximum(X - D, 0))
      _, key = random.split(key)
      return restock_counter, X, key

    # Use a for loop to perform the calculations on all states
    for i in range(sim_length):
        restock_counter, X, key = update_stock(
            restock_counter, X, key)
        
    return jnp.mean(restock_counter > 1, axis=0)
```

```{code-cell} ipython3
key = random.PRNGKey(1)
%time freq = compute_freq(key)
print(f"Frequency of at least two stock outs = {freq}")
```

Now let's write a `lax.scan` version that runs faster

```{code-cell} ipython3
@jax.jit
def compute_freq(key, x_init=70, sim_length=50, num_firms=1_000_000):
    # Prepare initial arrays
    X = jnp.full((num_firms, ), x_init)
    Z = random.normal(key, shape=(sim_length, num_firms))
    D = jnp.exp(mu + sigma * Z)

    # Stack the restock counter on top of the inventory
    restock_count = jnp.zeros((num_firms, ))
    Xs = jnp.vstack((X, restock_count))

    # Define the function for each update
    def update_X(Xs, D):

        # Separate the inventory and restock counter
        X = Xs[0]
        restock_count =  Xs[1]

        restock_count = jnp.where(X <= s,
                            restock_count + 1,
                            restock_count)
        X = jnp.where(X <= s, 
                      jnp.maximum(S - D, 0),
                      jnp.maximum(X - D, 0))
        
        Xs = jnp.vstack((X, restock_count))
        return Xs, None

    # Use lax.scan to perform the calculations on all states
    X_final, _ = lax.scan(update_X, Xs, D)

    return np.mean(X_final[1] > 1)
```

Note the time the routine takes to run, as well as the output.

```{code-cell} ipython3
%time freq = compute_freq(key)
print(f"Frequency of at least two stock outs = {freq}")
```
