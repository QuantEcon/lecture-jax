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


# Maximum Likelihood Estimation

```{contents} Contents
:depth: 2
```

```{include} _admonition/gpu.md
```

## Overview

This lecture is the extended JAX implementation of [this section](https://python.quantecon.org/mle.html#mle-with-numerical-methods) of [this lecture](https://python.quantecon.org/mle.html).

Please refer that lecture for all background and notation.

Here we will exploit the automatic differentiation capabilities of JAX rather than calculating derivatives by hand.

We'll require the following imports:

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  # set default figure size
from collections import namedtuple
import jax.numpy as jnp
import jax
from statsmodels.api import Poisson
```


Let's check the GPU we are running

```{code-cell} ipython3
!nvidia-smi
```


We will use 64 bit floats with JAX in order to increase the precision.

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```


## MLE with Numerical Methods (JAX)

Many distributions do not have nice, analytical solutions and therefore require
numerical methods to solve for parameter estimates.

One such numerical method is the Newton-Raphson algorithm.

Our goal is to find the maximum likelihood estimate $\hat{\boldsymbol{\beta}}$.

At $\hat{\boldsymbol{\beta}}$, the first derivative of the log-likelihood
function will be equal to 0.

Let's illustrate this by supposing

$$
\log \mathcal{L(\beta)} = - (\beta - 10) ^2 - 10
$$

Define the function `logL`.

```{code-cell} ipython3
@jax.jit
def logL(β):
    return -(β - 10) ** 2 - 10
```


To find the value of $\frac{d \log \mathcal{L(\boldsymbol{\beta})}}{d \boldsymbol{\beta}}$, we can use [jax.grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) which auto-differentiates the given function.

We further use [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) which vectorizes the given function i.e. the function acting upon scalar inputs can now be used with vector inputs.

```{code-cell} ipython3
dlogL = jax.vmap(jax.grad(logL))
```

```{code-cell} ipython3
β = jnp.linspace(1, 20)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))

ax1.plot(β, logL(β), lw=2)
ax2.plot(β, dlogL(β), lw=2)

ax1.set_ylabel(r'$log \mathcal{L(\beta)}$',
               rotation=0,
               labelpad=35,
               fontsize=15)
ax2.set_ylabel(r'$\frac{dlog \mathcal{L(\beta)}}{d \beta}$ ',
               rotation=0,
               labelpad=35,
               fontsize=19)

ax2.set_xlabel(r'$\beta$', fontsize=15)
ax1.grid(), ax2.grid()
plt.axhline(c='black')
plt.show()
```


The plot shows that the maximum likelihood value (the top plot) occurs
when $\frac{d \log \mathcal{L(\boldsymbol{\beta})}}{d \boldsymbol{\beta}} = 0$ (the bottom
plot).

Therefore, the likelihood is maximized when $\beta = 10$.

We can also ensure that this value is a *maximum* (as opposed to a
minimum) by checking that the second derivative (slope of the bottom
plot) is negative.

The Newton-Raphson algorithm finds a point where the first derivative is
0.

To use the algorithm, we take an initial guess at the maximum value,
$\beta_0$ (the OLS parameter estimates might be a reasonable
guess), then


Please refer to [this section](https://python.quantecon.org/mle.html#mle-with-numerical-methods) for the detailed algorithm.

Let's have a go at implementing the Newton-Raphson algorithm.

First, we'll create a `PoissonRegressionModel`.

```{code-cell} ipython3
PoissonRegressionModel = namedtuple('PoissonRegressionModel', ['X', 'y'])

def create_poisson_model(X, y):
    n, k = X.shape
    # Reshape y as a n_by_1 column vector
    y = y.reshape(n, 1)
    X, y = jax.device_put((X, y))
    return PoissonRegressionModel(X=X, y=y)
```


At present, JAX doesn't have an implementation to compute factorial directly.

In order to compute the factorial efficiently such that we can JIT it, we use

$$
    n! = e^{\log(\Gamma(n+1))}
$$

since [jax.lax.lgamma](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.lgamma.html) and [jax.lax.exp](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.exp.html) are available.

The following function `jax_factorial` computes the factorial using this idea.

```{code-cell} ipython3
@jax.jit
def _factorial(n):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0)).astype(int)

jax_factorial = jax.vmap(_factorial)
```


Let's define the Poisson Regression's log likelihood function.

```{code-cell} ipython3
@jax.jit
def poisson_logL(β, model):
    y = model.y
    μ = jnp.exp(model.X @ β)
    return jnp.sum(model.y * jnp.log(μ) - μ - jnp.log(jax_factorial(y)))
```


To find the gradient of the `poisson_logL`, we again use [jax.grad](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html).

According to [the documentation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobians-and-hessians-using-jacfwd-and-jacrev),

* `jax.jacfwd` uses forward-mode automatic differentiation, which is more efficient for “tall” Jacobian matrices, while
* `jax.jacrev` uses reverse-mode, which is more efficient for “wide” Jacobian matrices.

(The documentation also states that when matrices that are near-square, `jax.jacfwd` probably has an edge over `jax.jacrev`.)

Therefore, to find the Hessian, we can directly use `jax.jacfwd`.

```{code-cell} ipython3
G_poisson_logL = jax.grad(poisson_logL)
H_poisson_logL = jax.jacfwd(G_poisson_logL)
```


Our function `newton_raphson` will take a `PoissonRegressionModel` object
that has an initial guess of the parameter vector $\boldsymbol{\beta}_0$.

The algorithm will update the parameter vector according to the updating
rule, and recalculate the gradient and Hessian matrices at the new
parameter estimates.

```{code-cell} ipython3
def newton_raphson(model, β, tol=1e-3, max_iter=100, display=True):

    i = 0
    error = 100  # Initial error value

    # Print header of output
    if display:
        header = f'{"Iteration_k":<13}{"Log-likelihood":<16}{"θ":<60}'
        print(header)
        print("-" * len(header))

    # While loop runs while any value in error is greater
    # than the tolerance until max iterations are reached
    while jnp.any(error > tol) and i < max_iter:
        H, G = jnp.squeeze(H_poisson_logL(β, model)), G_poisson_logL(β, model)
        β_new = β - (jnp.dot(jnp.linalg.inv(H), G))
        error = jnp.abs(β_new - β)
        β = β_new

        if display:
            β_list = [f'{t:.3}' for t in list(β.flatten())]
            update = f'{i:<13}{poisson_logL(β, model):<16.8}{β_list}'
            print(update)

        i += 1

    print(f'Number of iterations: {i}')
    print(f'β_hat = {β.flatten()}')

    return β
```


Let's try out our algorithm with a small dataset of 5 observations and 3
variables in $\mathbf{X}$.

```{code-cell} ipython3
X = jnp.array([[1, 2, 5],
               [1, 1, 3],
               [1, 4, 2],
               [1, 5, 2],
               [1, 3, 1]])

y = jnp.array([1, 0, 1, 1, 0])

# Take a guess at initial βs
init_β = jnp.array([0.1, 0.1, 0.1]).reshape(X.shape[1], 1)

# Create an object with Poisson model values
poi = create_poisson_model(X, y)

# Use newton_raphson to find the MLE
β_hat = newton_raphson(poi, init_β, display=True)
```


As this was a simple model with few observations, the algorithm achieved
convergence in only 7 iterations.

The gradient vector should be close to 0 at $\hat{\boldsymbol{\beta}}$

```{code-cell} ipython3
G_poisson_logL(β_hat, poi)
```


## MLE with `statsmodels`

We’ll use the Poisson regression model in `statsmodels` to verify the results
obtained using JAX.

`statsmodels` uses the same algorithm as above to find the maximum
likelihood estimates.

Now, as `statsmodels` accepts only NumPy arrays, we can use the `__array__` method
of JAX arrays to convert it to NumPy arrays.

```{code-cell} ipython3
X_numpy = X.__array__()
y_numpy = y.__array__()
```

```{code-cell} ipython3
stats_poisson = Poisson(y_numpy, X_numpy).fit()
print(stats_poisson.summary())
```


```{exercise-start}
:label: newton_mle1
```

Define a quadratic model for a single, continuous explanatory variable, given by,

$$
    \log(\lambda_t) = \beta_0 + \beta_1 x_t + \beta_2 x_{t}^2
$$

Calculate the mean on the original scale instead of the log scale by exponentiating both sides of the above equation, which gives,

```{math}
:label: lambda_mle
    \lambda_t = \exp(\beta_0 + \beta_1 x_t + \beta_2 x_{t}^2)
```

Simulate the values of $x_t$ and $\lambda_t$ by using the following constants:

$$
    \beta_0 = -2.5,\\
    \beta_1 = 0.25, \\
    \beta_2 = 0.5
$$

Try to obtain the approximate values of $\beta_0,\beta_1,\beta_2$, by simulating a Poission Regression Model such that

$$
      y_t \sim Poisson(\lambda_t)
$$

By using sufficient large sample sizes, you should get the approximate values of $\beta_0,\beta_1,\beta_2$ by using `newton_raphson` function on the $X = [1, x_t, x_t^{2}]$ and $y_t$.


```{exercise-end}
```

```{solution-start} newton_mle1
:class: dropdown
```

Let's start by defining the constants.

```{code-cell} ipython3
b0 = -2.5
b1 = 0.25
b2 = 0.5
```


To simulate the model, we sample 500,000 values of $x_t$ from the uniform distribution.

```{code-cell} ipython3
seed = 32
shape = (500_000, 1)
key = jax.random.PRNGKey(seed)
x = jax.random.uniform(key, shape)
```


Compute $\lambda$ using {eq}`lambda_mle`

```{code-cell} ipython3
λ = jnp.exp(b0 + b1*x + b2*x**2)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 8))

ax1.scatter(x, λ, lw=2)
ax2.scatter(x, jnp.log(λ), lw=2)

ax1.set_ylabel(r'$\lambda$',
               rotation=0,
               labelpad=20,
               fontsize=15)
ax2.set_ylabel(r'$\log(\lambda)$ ',
               rotation=0,
               labelpad=20,
               fontsize=15)

ax2.set_xlabel(r'$x$', fontsize=15)
ax1.grid(), ax2.grid()
plt.axhline(c='black')
plt.show()
```


Let's define $y_t$ by sampling from a poission distribution with mean as $\lambda_t$.

This adds Poission error to the mean.

```{code-cell} ipython3
y = jax.random.poisson(key, λ, shape)
```

```{code-cell} ipython3
X = jnp.hstack((jnp.ones(shape), x, x**2))

# Take a guess at initial βs
init_β = jnp.array([0.1, 0.1, 0.1]).reshape(X.shape[1], 1)

# Create an object with Poisson model values
poi = create_poisson_model(X, poiss_y)

# Use newton_raphson to find the MLE
β_hat = newton_raphson(poi, init_β, tol=1e-6, display=True)
```


We obtain $\beta_0 \approx -2.5, \beta_1 \approx 0.287,$ and $\beta_2 \approx 0.455$ after 7 iterations.

```{solution-end}
```
