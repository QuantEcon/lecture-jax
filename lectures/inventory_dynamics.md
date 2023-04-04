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
from numba import njit, float64, prange
from numba.experimental import jitclass
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

Here's a class that stores parameters and generates time paths for inventory.

```{code-cell} python3
firm_data = [
   ('s', float64),          # restock trigger level
   ('S', float64),          # capacity
   ('mu', float64),         # shock location parameter
   ('sigma', float64)       # shock scale parameter
]


@jitclass(firm_data)
class Firm:

    def __init__(self, s=10, S=100, mu=1.0, sigma=0.5):

        self.s, self.S, self.mu, self.sigma = s, S, mu, sigma

    def update(self, x):
        "Update the state from t to t+1 given current state x."

        Z = np.random.randn()
        D = np.exp(self.mu + self.sigma * Z)
        if x <= self.s:
            return max(self.S - D, 0)
        else:
            return max(x - D, 0)

    def sim_inventory_path(self, x_init, sim_length):

        X = np.empty(sim_length)
        X[0] = x_init

        for t in range(sim_length-1):
            X[t+1] = self.update(X[t])
        return X
```

Let's run a first simulation, of a single path:

```{code-cell} ipython3
firm = Firm()

s, S = firm.s, firm.S
sim_length = 100
x_init = 50

X = firm.sim_inventory_path(x_init, sim_length)

fig, ax = plt.subplots()
bbox = (0., 1.02, 1., .102)
legend_args = {'ncol': 3,
               'bbox_to_anchor': bbox,
               'loc': 3,
               'mode': 'expand'}

ax.plot(X, label="inventory")
ax.plot(np.full(sim_length, s), 'k--', label="$s$")
ax.plot(np.full(sim_length, S), 'k-', label="$S$")
ax.set_ylim(0, S+10)
ax.set_xlabel("time")
ax.legend(**legend_args)

plt.show()
```

Now let's simulate multiple paths in order to build a more complete picture of
the probabilities of different outcomes:

```{code-cell} ipython3
sim_length=200
fig, ax = plt.subplots()

ax.plot(np.full(sim_length, s), 'k--', label="$s$")
ax.plot(np.full(sim_length, S), 'k-', label="$S$")
ax.set_ylim(0, S+10)
ax.legend(**legend_args)

for i in range(400):
    X = firm.sim_inventory_path(x_init, sim_length)
    ax.plot(X, 'b', alpha=0.2, lw=0.5)

plt.show()
```


## Marginal Distributions

Now let’s look at the marginal distribution $\psi_T$ of $X_T$ for some
fixed $T$.

We can also approximate the distribution using a [kernel density estimator](https://en.wikipedia.org/wiki/Kernel_density_estimation).

Kernel density estimators can be thought of as smoothed histograms.

They are preferable to histograms when the distribution being estimated is likely to be smooth.

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

## Exercises

```{exercise}
:label: id_ex1

This model is asymptotically stationary, with a unique stationary
distribution.

(See the discussion of stationarity in [our lecture on AR(1) processes](https://python.quantecon.org/ar1_processes.html) for background --- the fundamental concepts are the same.)

In particular, the sequence of marginal distributions $\{\psi_t\}$
is converging to a unique limiting distribution that does not depend on
initial conditions.

Although we will not prove this here, we can investigate it using simulation.

Your task is to generate and plot the sequence $\{\psi_t\}$ at times
$t = 10, 50, 250, 500, 750$ based on the discussion above.

(The kernel density estimator is probably the best way to present each
distribution.)

You should see convergence, in the sense that differences between successive distributions are getting smaller.

Try different initial conditions to verify that, in the long run, the distribution is invariant across initial conditions.
```

```{solution-start} id_ex1
:class: dropdown
```

Below is one possible solution:

The computations involve a lot of CPU cycles so we have tried to write the
code efficiently.

This meant writing a specialized function rather than using the class above.

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

```{code-cell} ipython3
x_init = 50.0
sample_dates = 10, 50, 250, 500, 750
s, S, mu, sigma = firm.s, firm.S, firm.mu, firm.sigma

fig, ax = plt.subplots()

key = random.PRNGKey(10)

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

You can convince yourself that initial conditions don’t matter by
testing a few of them.

For example, try rerunning the code above will all firms starting at
$X_0 = 20$ or $X_0 = 80$.

```{solution-end}
```

```{exercise}
:label: id_ex2

Using simulation, calculate the probability that firms that start with
$X_0 = 70$ need to order twice or more in the first 50 periods.

You will need a large sample size to get an accurate reading.
```


```{solution-start} id_ex2
:class: dropdown
```

Here is one solution.

Again, the computations are relatively intensive so we have written a a
specialized function rather than using the class above.

We will also use parallelization across firms.

```{code-cell} ipython3
@jax.jit
def compute_freq(key, x_init=70, sim_length=50, num_firms=1_000_000):
    # Prepare initial arrays
    X = jnp.full((num_firms, ), x_init)
    Z = random.normal(key, shape=(sim_length, num_firms))
    D = jnp.exp(mu + sigma * Z)

    # Stack the restock counter on top of the inventory
    restock_counter = jnp.zeros((num_firms, ))
    Xs = jnp.concatenate((X, restock_counter), axis=0)
    
    # Define the function for each update
    def update_X(Xs, D):

        # Separate the inventory and restock counter
        X = Xs[:num_firms]
        restock_counter =  Xs[num_firms:]
        X = jnp.where(X <= s, 
                      jnp.maximum(S - D, 0),
                      jnp.maximum(X - D, 0))
        restock_counter = jnp.where(X <= s,
                                    restock_counter + 1,
                                    restock_counter)
        Xs = jnp.concatenate((X, restock_counter), axis=0)
        return Xs, Xs

    # Use lax.scan to perform the calculations on all states
    _, X_final = lax.scan(update_X, Xs, D)
    firm_count = jnp.any(X_final[:, num_firms:] > 1, axis=0)
    return np.mean(firm_count)
```

Note the time the routine takes to run, as well as the output.

```{code-cell} ipython3
key = random.PRNGKey(1)
%time freq = compute_freq(key)
print(f"Frequency of at least two stock outs = {freq}")
```

```{solution-end}
```
