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

+++ {"user_expressions": []}

```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Kesten Processes and Firm Dynamics

```{index} single: Linear State Space Models
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

+++ {"user_expressions": []}

## Overview

Let's start with some imports:

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import quantecon as qe
import jax
import jax.numpy as jnp
from jax import random

# Check if JAX is using GPU
print(f"jax backend: {jax.devices()[0].platform}")
```

+++ {"user_expressions": []}

## Kesten Processes

```{index} single: Kesten processes; heavy tails
```

A **Kesten process** is a stochastic process of the form

```{math}
:label: kesproc

X_{t+1} = a_{t+1} X_t + \eta_{t+1}
```

where $\{a_t\}_{t \geq 1}$ and $\{\eta_t\}_{t \geq 1}$ are IID
sequences.

We are interested in the dynamics of $\{X_t\}_{t \geq 0}$ when $X_0$ is given.

We will focus on the nonnegative scalar case, where $X_t$ takes values in $\mathbb R_+$.

In particular, we will assume that

* the initial condition $X_0$ is nonnegative,
* $\{a_t\}_{t \geq 1}$ is a nonnegative IID stochastic process and
* $\{\eta_t\}_{t \geq 1}$ is another nonnegative IID stochastic process, independent of the first.

+++ {"user_expressions": []}

## Application: Firm Dynamics

### Gibrat's Law

It was postulated many years ago by Robert Gibrat {cite}`gibrat1931inegalites` that firm size evolves according to a simple rule whereby size next period is proportional to current size.

This is now known as [Gibrat's law of proportional growth](https://en.wikipedia.org/wiki/Gibrat%27s_law).

We can express this idea by stating that a suitably defined measure
$s_t$ of firm size obeys

```{math}
:label: firm_dynam_gb

\frac{s_{t+1}}{s_t} = a_{t+1}
```

for some positive IID sequence $\{a_t\}$.

We can accommodate empirical findings by modifying {eq}`firm_dynam_gb`
to

```{math}
:label: firm_dynam

s_{t+1} = a_{t+1} s_t + b_{t+1}
```

where $\{a_t\}$ and $\{b_t\}$ are both IID and independent of each
other.

In the exercises you are asked to show that {eq}`firm_dynam` is more
consistent with the empirical findings presented above than Gibrat's law in
{eq}`firm_dynam_gb`.

### Heavy Tails

If the conditions of the Kesten--Goldie Theorem are satisfied, then the firm
size distribution is predicted to have heavy tails.

Now we explore this idea further, generalizing the firm
size dynamics and examining the corresponding rank-size plots.

One unrealistic aspect of the firm dynamics specified in {eq}`firm_dynam` is
that it ignores entry and exit.

In any given period and in any given market, we observe significant numbers of firms entering and exiting the market.

In this setting, firm dynamics can be expressed as

```{math}
:label: firm_dynam_ee

s_{t+1} = e_{t+1} \mathbb{1}\{s_t < \bar s\} +
(a_{t+1} s_t + b_{t+1}) \mathbb{1}\{s_t \geq \bar s\}
```

Here

* the state variable $s_t$ represents productivity (which is a proxy
  for output and hence firm size),
* the IID sequence $\{ e_t \}$ is thought of as a productivity draw for a new
  entrant and
* the variable $\bar s$ is a threshold value that we take as given.

The idea behind {eq}`firm_dynam_ee` is that firms stay in the market as long
as their productivity $s_t$ remains at or above $\bar s$.

* In this case, their productivity updates according to {eq}`firm_dynam`.

Firms choose to exit when their productivity $s_t$ falls below $\bar s$.

* In this case, they are replaced by a new firm with productivity
  $e_{t+1}$.

What can we say about dynamics?

Although {eq}`firm_dynam_ee` is not a Kesten process, it does update in the
same way as a Kesten process when $s_t$ is large.

So perhaps its stationary distribution still has Pareto tails?

We can investigate this question via simulation and rank-size plots.

The approach will be to

1. generate $M$ draws of $s_T$ when $M$ and $T$ are
   large and
1. plot the largest 1,000 of the resulting draws in a rank-size plot.

(The distribution of $s_T$ will be close to the stationary distribution
when $T$ is large.)

In the simulation, we assume that each of $a_t, b_t$ and $e_t$ is lognormal.

+++ {"user_expressions": []}

Now we can generate the observations with the following default parameters:

```{code-cell} ipython3
def generate_draws(M = 1_000_000,     # number of firms
                   μ_a = -0.5,        # location parameter for a
                   σ_a = 0.1,         # scale parameter for a
                   μ_b = 0.0,         # location parameter for b
                   σ_b = 0.5,         # scale parameter for b
                   μ_e = 0.0,         # location parameter for e
                   σ_e = 0.5,         # scale parameter for e
                   s_bar = 1.0,       # threshold
                   T = 500,           # sampling date
                   s_init = 1.0,      # initial condition for each firm
                   seed=123):

    key = random.PRNGKey(seed)
    keys = random.split(key, 3)

    # Initialize the array of s values with the initial value
    s = jnp.full((M, ), s_init)
    
    @jax.jit
    def update_s(s, keys):
        a_random = μ_a + σ_a * random.normal(keys[0], (M, ))
        b_random = μ_b + σ_b * random.normal(keys[1], (M, ))
        e_random = μ_e + σ_e * random.normal(keys[2], (M, ))

        exp_a = jnp.exp(a_random)
        exp_b = jnp.exp(b_random)
        exp_e = jnp.exp(e_random)

        s = jnp.where(s < s_bar,
                          exp_e,
                          exp_a * s + exp_b)

        return s, keys[-1]

    # Perform updates on s for time t
    for t in range(T):
        s, key = update_s(s, keys)
        keys = random.split(key, 3)

    return s

%time data = generate_draws().block_until_ready()
```

+++ {"user_expressions": []}

As JIT-compiled `for` loops will lead to very slow compilation, we used `jax.jit` on the function `update_s` instead of the whole function.

Let's produce the rank-size plot and check the distribution:

```{code-cell} ipython3
fig, ax = plt.subplots()

rank_data, size_data = qe.rank_size(data, c=0.01)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("log rank")
ax.set_ylabel("log size")

plt.show()
```

+++ {"user_expressions": []}

The plot produces a straight line, consistent with a Pareto tail.

It is possible to further speed up our code by replacing the `for` loop with [`lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)
to reduce the loop overhead in the compilation of the jitted function

```{code-cell} ipython3
from jax import lax

@jax.jit
def generate_draws_lax(μ_a=-0.5,
                       σ_a=0.1,
                       μ_b=0.0,
                       σ_b=0.5,
                       μ_e=0.0,
                       σ_e=0.5,
                       s_bar=1.0,
                       T=500,
                       M=1_000_000,
                       s_init=1.0,
                       seed=123):

    key = random.PRNGKey(seed)
    keys = random.split(key, T)

    # Generate random draws and initial values
    a_random = μ_a + σ_a * random.normal(keys[0], (T, M))
    b_random = μ_b + σ_b * random.normal(keys[1], (T, M))
    e_random = μ_e + σ_e * random.normal(keys[2], (T, M))
    s = jnp.full((M, ), s_init)

    # Define the function for each update
    def update_s(s, a_b_e_draws):
        a, b, e = a_b_e_draws
        s = jnp.where(s < s_bar,
                      jnp.exp(e),
                      jnp.exp(a) * s + jnp.exp(b))
        return s, None

    # Use lax.scan to perform the calculations on all states
    s_final, _ = lax.scan(update_s, s, (a_random, b_random, e_random))
    return s_final

%time data = generate_draws_lax().block_until_ready()
```

+++ {"user_expressions": []}

Since we used `jax.jit` on the entire function, the compiled function is even faster

```{code-cell} ipython3
%time data = generate_draws_lax().block_until_ready()
```

+++ {"user_expressions": []}

Here we produce the same rank-size plot:

```{code-cell} ipython3
fig, ax = plt.subplots()

rank_data, size_data = qe.rank_size(data, c=0.01)
ax.loglog(rank_data, size_data, 'o', markersize=3.0, alpha=0.5)
ax.set_xlabel("log rank")
ax.set_ylabel("log size")

plt.show()
```
