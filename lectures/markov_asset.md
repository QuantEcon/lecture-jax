# An Asset Pricing Problem

## Overview

In this lecture we consider a simple asset pricing problem and use it to
illustrate some foundations of JAX programming.

We will describe the underlying problem relatively briefly, so we can focus on
to the equation that needs to be solved.

Then we will show how to solve the problem using JAX.

Along the way we will make comments on how we structure coding problems when
working in the environment provided by JAX.

If you wish to skip all motivation and move straight to the equation we plan to
study, you can skip to [TODO add link]

Below we use the following imports

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from collections import namedtuple
```

We will use 64 bit floats with JAX in order to increase precision.

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

## Pricing a single payoff

Suppose, at time $t$, we have an asset that pays a random amount $G_{t+1}$ at
time $t+1$ and nothing after that.

The simplest way to price this asset is to use "risk-neutral" asset pricing, which
asserts that the price of the asset at time $t$ should be

$$
    P_t = \beta \mathbb E_t G_{t+1}
$$ (eq:rnp)

where $\beta$ is a discount factor and $\mathbb E_t G_{t+1}$ is the expectation
of $G_{t+1}$ at time $t$

Roughly speaking, this says that the cost (i.e., price) equals expected benefit.

The discount factor is introduced because most people prefer payments now to
payments in the future.

The problem with this very simple model is that it does not take into account
attitudes to risk.

For example, investors often demand higher rates of return for holding risky
assets.

This feature of asset prices cannot be captured by risk neutral pricing.

Hence we modify [](eq:rnp) to

$$
    P_t = \mathbb E_t M_{t+1} G_{t+1}
$$ (eq:nrnp)

In this expression, $M_{t+1}$ is called the **stochastic discount factor**.

In essence, allowing discounting to become a random variable gives us the
flexibilit to combine temporal discounting and attitudes to risk.

We omit further justification because our aim is to move to the computational problem.


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

V_t = {\mathbb E}_t \left[ M_{t+1} \exp(g_{d, t+1}) (1 + V_{t+1}) \right]
```

$$
    g_{d, t+1} = \ln \frac{D_{t+1}}{D_t}
$$

is the growth rate of dividends.

Our aim is to solve [](pdex2) but before that we need to specify

1. the stochastic discount factor $M_{t+1}$ and
1. the growth rate of dividends $g_{d, t+1}$


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
    g_{c, t+1} = \ln \left( \frac{C_{t+1}}{C_t} \right)
$$ 

the growth rate rate of consumption, we obtain 

```{math}
:label: lucsdf2
    M_{t+1}
    = \beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}
    = \beta \exp( g_{c, t+1} )^{-\gamma} 
    = \beta \exp(-\gamma g_{c, t+1})
```



## Solving for the price-dividend ratio

Substituting [](lucsdf2) into {eq}`pdex2` gives the price-dividend ratio
formula

$$
    V_t = \beta {\mathbb E}_t 
    \left[ \exp(g_{d, t+1} - \gamma g_{c, t+1}) (1 + V_{t+1}) \right]
$$ (pdex3)

For now we assume that there is a Markov chain $\{X_t\}$, which we call
the **state process**,  such that

$$
\begin{aligned}
    & g_{c, t+1} = \mu_c + X_t + \sigma_c \epsilon_{c, t+1} \\
    & g_{d, t+1} = \mu_d + X_t + \sigma_d \epsilon_{d, t+1} 
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

The unknown function $v$ satisfies the equation

$$
    v(X_t) = \beta {\mathbb E}_t 
    \left[
        \exp[
            a + (1-\gamma) X_t + 
                \sigma_c \epsilon_{c, t+1} - 
                \gamma  \sigma_d \epsilon_{d, t+1}     
            ]
        (1 + V(X_{t+1}))
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
$$ [](eq:ntecxv)

where $K$ is the matrix defined by 

$$
    K(x, y)
    = \beta \left[
        \exp[
            a + (1-\gamma) x + 
                (\sigma_c ^2 + \gamma^2  \sigma_d^2) / 2)
            ]
        (1 + v(y))
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


## Code

```{code-cell} python3
Model = namedtuple('Model',
    'n',    # size of state space for Markov chain
    'ρ',    # persistence parameter for Markov chain
    'β',    # discount factor
    'γ',    # coefficient of risk aversion

```

```{code-cell} python3
class AssetPriceModel:
    """
    A class that stores the primitives of the asset pricing model.

    Parameters
    ----------
    β : scalar, float
        Discount factor
    mc : MarkovChain
        Contains the transition matrix and set of state values for the state
        process
    γ : scalar(float)
        Coefficient of risk aversion
    g : callable
        The function mapping states to growth rates

    """
    def __init__(self, β=0.96, mc=None, γ=2.0, g=np.exp):
        self.β, self.γ = β, γ
        self.g = g

        # A default process for the Markov chain
        if mc is None:
            self.ρ = 0.9
            self.σ = 0.02
            self.mc = qe.tauchen(self.ρ, self.σ, n=25)
        else:
            self.mc = mc

        self.n = self.mc.P.shape[0]

    def test_stability(self, Q):
        """
        Stability test for a given matrix Q.
        """
        sr = np.max(np.abs(eigvals(Q)))
        if not sr < 1 / self.β:
            msg = f"Spectral radius condition failed with radius = {sr}"
            raise ValueError(msg)


def tree_price(ap):
    """
    Computes the price-dividend ratio of the Lucas tree.

    Parameters
    ----------
    ap: AssetPriceModel
        An instance of AssetPriceModel containing primitives

    Returns
    -------
    v : array_like(float)
        Lucas tree price-dividend ratio

    """
    # Simplify names, set up matrices
    β, γ, P, y = ap.β, ap.γ, ap.mc.P, ap.mc.state_values
    J = P * ap.g(y)**(1 - γ)

    # Make sure that a unique solution exists
    ap.test_stability(J)

    # Compute v
    I = np.identity(ap.n)
    Ones = np.ones(ap.n)
    v = solve(I - β * J, β * J @ Ones)

    return v
```

Here's a plot of $v$ as a function of the state for several values of $\gamma$,
with a positively correlated Markov process and $g(x) = \exp(x)$

```{code-cell} python3
γs = [1.2, 1.4, 1.6, 1.8, 2.0]
ap = AssetPriceModel()
states = ap.mc.state_values

fig, ax = plt.subplots()

for γ in γs:
    ap.γ = γ
    v = tree_price(ap)
    ax.plot(states, v, lw=2, alpha=0.6, label=rf"$\gamma = {γ}$")

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





## An Extended Example

We suppose that

$$
\begin{aligned}
    & g_{c, t+1} = \mu_c + Z_t + \exp(h_{c, t}) \epsilon_{c, t+1} \\
    & g_{d, t+1} = \mu_d + Z_t + \exp(h_{d, t}) \epsilon_{d, t+1} 
\end{aligned}
$$

where


Let $X_t = (h_{c, t}, h_{d, t}, Z_t)$.  


We call $\{X_t\}$ the state process and guess that $V_t$ is a function of
this state process --- and this guess turns out to be correct.

This means that $V_t = v(X_t)$ for some unknown function $v$.

The unknown function $v$ satisfies the equation


