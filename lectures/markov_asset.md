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

# An Asset Pricing Problem

## Overview

In this lecture we consider a simple asset pricing problem and use it to
illustrate some foundations of JAX programming.

If you wish to skip all motivation and move straight to the equation we plan to
solve, you can jump to [TODO add link]

Below we use the following imports

```{code-cell}
import quantecon as qe
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from collections import namedtuple
```

We will use 64 bit floats with JAX in order to increase precision.

```{code-cell}
jax.config.update("jax_enable_x64", True)
```

## Pricing a single payoff

Suppose, at time $t$, we have an asset that pays a random amount $D_{t+1}$ at
time $t+1$ and nothing after that.

The simplest way to price this asset is to use "risk-neutral" asset pricing, which
asserts that the price of the asset at time $t$ should be

$$
    P_t = \beta \mathbb E_t D_{t+1}
$$ (eq:rnp)

where $\beta$ is a constant discount factor and $\mathbb E_t D_{t+1}$ is the expectation
of $D_{t+1}$ at time $t$.

Roughly speaking, [](eq:rnp) says that the cost (i.e., price) equals expected benefit.

The discount factor is introduced because most people prefer payments now to
payments in the future.

One problem with this very simple model is that it does not take into account
attitudes to risk.

For example, investors often demand higher rates of return for holding risky
assets.

This feature of asset prices cannot be captured by risk neutral pricing.

Hence we modify [](eq:rnp) to

$$
    P_t = \mathbb E_t M_{t+1} D_{t+1}
$$ (eq:nrnp)

In this expression, $M_{t+1}$ replaces $\beta$ and is called the **stochastic discount factor**.

In essence, allowing discounting to become a random variable gives us the
flexibility to combine temporal discounting and attitudes to risk.

We leave further discussion to [other lectures](https://python.quantecon.org/markov_asset.html) 
because our aim is to move to the computational problem.

+++

## Pricing a cash flow

Now let's try to price an asset like a share, which delivers a cash flow $D_t,
D_{t+1}, \ldots$.

We will call these payoffs "dividends".

If we buy the share, hold it for one period and sell it again, we receive one
dividend and our payoff is $D_{t+1} + P_{t+1}$.

Therefore, by [](eq:nrnp), the price should be

$$
    P_t = \mathbb E_t M_{t+1} [ D_{t+1} + P_{t+1} ]
$$ (lteeqs0)

Because prices generally grow over time, which complicates analysis, it will be
easier for us to solve for the **price-dividend ratio** $V_t := P_t / D_t$.

Let's write down an expression that this ratio should satisfy.

We can divide both sides of {eq}`lteeqs0` by $D_t$ to get

```{math}
:label: pdex

V_t = {\mathbb E}_t \left[ M_{t+1} \frac{D_{t+1}}{D_t} (1 + V_{t+1}) \right]
```

We can also write this as

```{math}
:label: pdex2

V_t = {\mathbb E}_t \left[ M_{t+1} \exp(G^d_{t+1}) (1 + V_{t+1}) \right]
```

$$
    G^d_{t+1} = \ln \frac{D_{t+1}}{D_t}
$$

is the growth rate of dividends.

Our aim is to solve [](pdex2) but before that we need to specify

1. the stochastic discount factor $M_{t+1}$ and
1. the growth rate of dividends $G^d_{t+1}$

+++

## Choosing the stochastic discount factor

We will adopt the stochastic discount factor described in {cite}`Lucas1978`, which has the form

```{math}
:label: lucsdf
    M_{t+1} = \beta \frac{u'(C_{t+1})}{u'(C_t)}
```

where $u$ is a utility function and $C_t$ is time $t$ consumption of a representative consumer.

(An explanation of the ideas behind this expression is given in [a later lecture](https://python-advanced.quantecon.org/lucas_model.html) and we omit further details and motivation.)

For utility, we'll assume the **constant relative risk aversion** (CRRA) specification

```{math}
:label: eqCRRA
    u(c) = \frac{c^{1-\gamma}}{1 - \gamma} 
```

Inserting the CRRA specification into {eq}`lucsdf` and letting

$$
    G^c_{t+1} = \ln \left( \frac{C_{t+1}}{C_t} \right)
$$ 

the growth rate rate of consumption, we obtain 

```{math}
:label: lucsdf2
    M_{t+1}
    = \beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}
    = \beta \exp( G^c_{t+1} )^{-\gamma} 
    = \beta \exp(-\gamma G^c_{t+1})
```

+++

## Solving for the price-dividend ratio

Substituting [](lucsdf2) into {eq}`pdex2` gives the price-dividend ratio
formula

$$
    V_t = \beta {\mathbb E}_t 
    \left[ \exp(G^d_{t+1} - \gamma G^c_{t+1}) (1 + V_{t+1}) \right]
$$ (pdex3)

For now we assume that there is a Markov chain $\{X_t\}$, which we call
the **state process**,  such that

$$
\begin{aligned}
    & G^c_{t+1} = \mu_c + X_t + \sigma_c \epsilon_{c, t+1} \\
    & G^d_{t+1} = \mu_d + X_t + \sigma_d \epsilon_{d, t+1} 
\end{aligned}
$$

Here $\{\epsilon_{c, t}\}$ and $\{\epsilon_{d, t}\}$ are IID and standard
normal, and independent of eachother.

We can think of $\{X_t\}$ as an aggregate shock that affects both consumption
growth and firm profits (and hence dividends).

We let $P$ be the [stochastic matrix that governs $\{X_t\}$](https://python.quantecon.org/finite_markov.html) and assume $\{X_t\}$ takes values in some finite set $S$.

We guess that $V_t$ is a function of this state process (and this guess turns
out to be correct).

This means that $V_t = v(X_t)$ for some unknown function $v$.

By [](pdex3), the unknown function $v$ satisfies the equation

$$
    v(X_t) = \beta {\mathbb E}_t 
    \left[
        \exp[
            a + (1-\gamma) X_t + 
                \sigma_c \epsilon_{c, t+1} - 
                \gamma  \sigma_d \epsilon_{d, t+1}     
            ]
        (1 + v(X_{t+1}))
    \right]
$$ (eq:neweqn101)

where $a := \mu_c - \gamma \mu_d$

Since the shocks $\epsilon_{c, t+1}$ and $\epsilon_{d, t+1}$ are independent of
$\{X_t\}$, we can integrate them out.

We use the following property of lognormal distributions: if $Y = \exp(c
\epsilon)$ for constant $c$ and $\epsilon \sim N(0,1)$, then $\mathbb E Y =
\exp(c^2/2)$.

This yields

$$
    v(X_t) = \beta {\mathbb E}_t 
    \left[
        \exp[
            a + (1-\gamma) X_t + 
                (\sigma_c ^2 + \gamma^2  \sigma_d^2) / 2)
            ]
        (1 + v(X_{t+1}))
    \right]
$$ (eq:ntev)

Conditioning on $X_t = x$, we can write this as

$$
    v(x) = \beta \sum_{y \in S}
    \left[
        \exp[
            a + (1-\gamma) x + 
                (\sigma_c ^2 + \gamma^2  \sigma_d^2) / 2)
            ]
        (1 + v(y))
    \right] P(x, y)
$$ (eq:ntecx)

for all $x \in S$.

Suppose $S = \{x_1, \ldots, x_n\}$.

Then we can think of $v(x_1), \ldots, v(x_n)$ as an $n \times 1$ vector.

We can write [](eq:ntecx) in vector form as

$$
    v = K (\mathbb 1 + v)
$$ (eq:ntecxv)

where $K$ is the matrix defined by 

$$
    K(x, y)
    = \beta \left[
        \exp[
            a + (1-\gamma) x + 
                (\sigma_c ^2 + \gamma^2  \sigma_d^2) / 2)
            ]
    \right] P(x, y)
$$

(That, is $K$ is a matrix with $i,j$-th element $K(x_i, x_j)$.)

Notice that [](eq:ntecxv) can be written as $(I - K)v = K \mathbb 1$.

The Neumann series lemma tells us that $(I - K)$ is invertible and the solution
is

$$
    v = (I - K)^{-1} K \mathbb 1
$$ (eq:ntecxvv)

whenever $r(K)$, the spectral radius of $K$, is strictly less than one.

Once we specify $P$ and all the parameters, we can obtain $K$ and 
then compute the solution [](eq:ntecxvv).

+++

## Code

```{code-cell}
Model = namedtuple('Model', 
                   ('P', 'S', 'β', 'γ', 'μ_c', 'μ_d', 'σ_c', 'σ_d'))

def create_model(n=100,         # size of state space for Markov chain
                 ρ=0.2,         # persistence parameter for Markov chain
                 σ=0.1,         # persistence parameter for Markov chain
                 β=0.98,        # discount factor
                 γ=2.5,         # coefficient of risk aversion 
                 μ_c=0.01,      # mean growth of consumtion
                 μ_d=0.01,      # mean growth of dividends
                 σ_c=0.02,      # consumption volatility 
                 σ_d=0.04):     # dividend volatility 

    mc = qe.tauchen(n, ρ, σ, 0)
    S = mc.state_values
    P = mc.P
    return Model(P=P, S=S, β=β, γ=γ, μ_c=μ_c, μ_d=μ_d, σ_c=σ_c, σ_d=σ_d)


def test_stability(Q):
    """
    Stability test for a given matrix Q.
    """
    sr = np.max(np.abs(np.linalg.eigvals(Q)))
    if not sr < 1:
        msg = f"Spectral radius condition failed with radius = {sr}"
        raise ValueError(msg)

def compute_K(model):
    # Setp up
    P, S, β, γ, μ_c, μ_d, σ_c, σ_d = model
    n = len(S)
    # Reshape and multiply pointwise using broadcasting
    x = np.reshape(S, (n, 1))
    a = μ_c - γ * μ_d
    e = np.exp(a + (1 - γ) * x + (σ_c**2 + γ**2 * σ_d**2) / 2)
    K = β * e * P
    return K

def compute_K_loop(model):
    # Setp up
    P, S, β, γ, μ_c, μ_d, σ_c, σ_d = model
    n = len(S)
    K = np.empty((n, n))
    a = μ_c - γ * μ_d
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            e = np.exp(a + (1 - γ) * x + (σ_c**2 + γ**2 * σ_d**2) / 2)
            K[i, j] = β * e * P[i, j]
    return K

def price_dividend_ratio(model):
    """
    Computes the price-dividend ratio of the asset.

    Parameters
    ----------
    model: an instance of Model
        contains primitives

    Returns
    -------
    v : array_like
        price-dividend ratio

    """
    K = compute_K(model)
    n = len(model.S)
    # Make sure that a unique solution exists
    test_stability(K)

    # Compute v
    I = np.identity(n)
    Ones = np.ones(n)
    v = np.linalg.solve(I - K, K @ Ones)

    return v
```

Here's a plot of $v$ as a function of the state for several values of $\gamma$,
with a positively correlated Markov process and $g(x) = \exp(x)$

```{code-cell}
model = create_model()
S = model.S
γs = np.linspace(2.0, 3.0, 5)

fig, ax = plt.subplots()

for γ in γs:
    model = create_model(γ=γ)
    v = price_dividend_ratio(model)
    ax.plot(S, v, lw=2, alpha=0.6, label=rf"$\gamma = {γ}$")

ax.set_title('Price-divdend ratio as a function of the state')
ax.set_ylabel("price-dividend ratio")
ax.set_xlabel("state")
ax.legend(loc='upper right')
plt.show()
```

Notice that $v$ is decreasing in each case.

This is because, with a positively correlated state process, higher states indicate higher future consumption growth.

With the stochastic discount factor {eq}`lucsdf2`, higher growth decreases the
discount factor, lowering the weight placed on future dividends.

+++

## An Extended Example

We suppose that

$$
\begin{aligned}
    & G^c_{t+1} = \mu_c + Z_t + \exp(h_{c, t}) \epsilon_{c, t+1} \\
    & G^d_{t+1} = \mu_d + Z_t + \exp(h_{d, t}) \epsilon_{d, t+1} 
\end{aligned}
$$

where

+++

Let $X_t = (h_{c, t}, h_{d, t}, Z_t)$.

+++

We call $\{X_t\}$ the state process and guess that $V_t$ is a function of
this state process --- and this guess turns out to be correct.

This means that $V_t = v(X_t)$ for some unknown function $v$.

The unknown function $v$ satisfies the equation
