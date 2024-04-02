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

# Job Search

```{include} _admonition/gpu.md
```


In this lecture we study a basic infinite-horizon job search with Markov wage
draws 

The exercise at the end asks you to add recursive preferences and compare
the result.

In addition to what’s in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install quantecon
```

We use the following imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import quantecon as qe
import jax
import jax.numpy as jnp
from collections import namedtuple

jax.config.update("jax_enable_x64", True)
```

## Model

We study an elementary model where 

* jobs are permanent 
* unemployed workers receive current compensation $c$
* the wage offer distribution $\{W_t\}$ is Markovian
* the horizon is infinite
* an unemployment agent discounts the future via discount factor $\beta \in (0,1)$

The wage process obeys

$$
    W_{t+1} = \rho W_t + \nu Z_{t+1},
    \qquad \{Z_t\} \text{ is IID and } N(0, 1)
$$

We discretize this using Tauchen's method to produce a stochastic matrix $P$

Since jobs are permanent, the return to accepting wage offer $w$ today is

$$
    w + \beta w + \beta^2 w + \frac{w}{1-\beta}
$$

The Bellman equation is

$$
    v(w) = \max
    \left\{
            \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
    \right\}
$$

We solve this model using value function iteration.


Let's set up a namedtuple to store information needed to solve the model.

```{code-cell} ipython3
Model = namedtuple('Model', ('n', 'w_vals', 'P', 'β', 'c', 'θ'))
```

The function below holds default values and populates the namedtuple.

```{code-cell} ipython3
def create_js_model(
        n=500,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.99,      # discount factor
        c=1.0,       # unemployment compensation
        θ=-0.1       # risk parameter
    ):
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(mc.state_values), mc.P
    P = jnp.array(P)
    return Model(n, w_vals, P, β, c, θ)
```

Here's the Bellman operator.

```{code-cell} ipython3
@jax.jit
def T(v, model):
    """
    The Bellman operator Tv = max{e, c + β E v} with 

        e(w) = w / (1-β) and (Ev)(w) = E_w[ v(W')]

    """
    n, w_vals, P, β, c, θ = model
    h = c + β * P @ v
    e = w_vals / (1 - β)

    return jnp.maximum(e, h)
```

The next function computes the optimal policy under the assumption that $v$ is
                 the value function.

The policy takes the form

$$
    \sigma(w) = \mathbf 1 
        \left\{
            \frac{w}{1-\beta} \geq c + \beta \sum_{w'} v(w') P(w, w')
        \right\}
$$

Here $\mathbf 1$ is an indicator function.

The statement above means that the worker accepts ($\sigma(w) = 1$) when the value of stopping
is higher than the value of continuing.

```{code-cell} ipython3
@jax.jit
def get_greedy(v, model):
    """Get a v-greedy policy."""
    n, w_vals, P, β, c, θ = model
    e = w_vals / (1 - β)
    h = c + β * P @ v
    σ = jnp.where(e >= h, 1, 0)
    return σ
```

Here's a routine for value function iteration.

```{code-cell} ipython3
def vfi(model, max_iter=10_000, tol=1e-4):
    """Solve the infinite-horizon Markov job search model by VFI."""

    print("Starting VFI iteration.")
    v = jnp.zeros_like(model.w_vals)    # Initial guess
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        new_v = T(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        i += 1
        v = new_v

    v_star = v
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
```

### Computing the solution

Let's set up and solve the model.

```{code-cell} ipython3
model = create_js_model()
n, w_vals, P, β, c, θ = model

qe.tic()
v_star, σ_star = vfi(model)
vfi_time = qe.toc()
```

We compute the reservation wage as the first $w$ such that $\sigma(w)=1$.

```{code-cell} ipython3
res_wage = w_vals[jnp.searchsorted(σ_star, 1.0)]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, v_star, alpha=0.8, label="value function")
ax.vlines((res_wage,), 150, 400, 'k', ls='--', label="reservation wage")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

## Exercise

```{exercise-start}
:label: job_search_1
```

In the setting above, the agent is risk-neutral vis-a-vis future utility risk.

Now solve the same problem but this time assuming that the agent has risk-sensitive
preferences, which are a type of nonlinear recursive preferences.

The Bellman equation becomes

$$
    v(w) = \max
    \left\{
            \frac{w}{1-\beta}, 
            c + \frac{\beta}{\theta}
            \ln \left[ 
                      \sum_{w'} \exp(\theta v(w')) P(w, w')
                \right]
    \right\}
$$


When $\theta < 0$ the agent is risk sensitive.

Solve the model when $\theta = -0.1$ and compare your result to the risk neutral
case.

Try to interpret your result.

```{exercise-end}
```

```{solution-start} job_search_1
:class: dropdown
```

```{code-cell} ipython3
def create_risk_sensitive_js_model(
        n=500,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.99,      # discount factor
        c=1.0,       # unemployment compensation
        θ=-0.1       # risk parameter
    ):
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(mc.state_values), mc.P
    P = jnp.array(P)
    return Model(n, w_vals, P, β, c, θ)


@jax.jit
def T_rs(v, model):
    """
    The Bellman operator Tv = max{e, c + β R v} with 

        e(w) = w / (1-β) and

        (Rv)(w) = (1/θ) ln{E_w[ exp(θ v(W'))]}

    """
    n, w_vals, P, β, c, θ = model
    h = c + (β / θ) * jnp.log(P @ (jnp.exp(θ * v)))
    e = w_vals / (1 - β)

    return jnp.maximum(e, h)


@jax.jit
def get_greedy_rs(v, model):
    " Get a v-greedy policy."
    n, w_vals, P, β, c, θ = model
    e = w_vals / (1 - β)
    h = c + (β / θ) * jnp.log(P @ (jnp.exp(θ * v)))
    σ = jnp.where(e >= h, 1, 0)
    return σ



def vfi(model, max_iter=10_000, tol=1e-4):
    "Solve the infinite-horizon Markov job search model by VFI."
    print("Starting VFI iteration.")
    v = jnp.zeros_like(model.w_vals)    # Initial guess
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        new_v = T_rs(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        i += 1
        v = new_v

    v_star = v
    σ_star = get_greedy_rs(v_star, model)
    return v_star, σ_star



model_rs = create_risk_sensitive_js_model()

n, w_vals, P, β, c, θ = model_rs

qe.tic()
v_star_rs, σ_star_rs = vfi(model_rs)
vfi_time = qe.toc()
```

```{code-cell} ipython3
res_wage_rs = w_vals[jnp.searchsorted(σ_star_rs, 1.0)]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, v_star,  alpha=0.8, label="RN $v$")
ax.plot(w_vals, v_star_rs, alpha=0.8, label="RS $v$")
ax.vlines((res_wage,), 150, 400,  ls='--', color='darkblue', alpha=0.5, label=r"RV $\bar w$")
ax.vlines((res_wage_rs,), 150, 400, ls='--', color='orange', alpha=0.5, label=r"RS $\bar w$")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

```{solution-end}
```