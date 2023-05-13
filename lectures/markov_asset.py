# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # An Asset Pricing Problem
#
# ## Overview
#
# In this lecture we consider a simple asset pricing problem and use it to
# illustrate some foundations of JAX programming.
#
# If you wish to skip all motivation and move straight to the equation we plan to
# solve, you can jump to [TODO add link]
#
# Below we use the following imports

import quantecon as qe
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from collections import namedtuple

# We will use 64 bit floats with JAX in order to increase precision.

jax.config.update("jax_enable_x64", True)

# ## Pricing a single payoff
#
# Suppose, at time $t$, we have an asset that pays a random amount $D_{t+1}$ at
# time $t+1$ and nothing after that.
#
# The simplest way to price this asset is to use "risk-neutral" asset pricing, which
# asserts that the price of the asset at time $t$ should be
#
# $$
#     P_t = \beta \mathbb E_t D_{t+1}
# $$ (eq:rnp)
#
# where $\beta$ is a constant discount factor and $\mathbb E_t D_{t+1}$ is the expectation
# of $D_{t+1}$ at time $t$.
#
# Roughly speaking, [](eq:rnp) says that the cost (i.e., price) equals expected benefit.
#
# The discount factor is introduced because most people prefer payments now to
# payments in the future.
#
# One problem with this very simple model is that it does not take into account
# attitudes to risk.
#
# For example, investors often demand higher rates of return for holding risky
# assets.
#
# This feature of asset prices cannot be captured by risk neutral pricing.
#
# Hence we modify [](eq:rnp) to
#
# $$
#     P_t = \mathbb E_t M_{t+1} D_{t+1}
# $$ (eq:nrnp)
#
# In this expression, $M_{t+1}$ replaces $\beta$ and is called the **stochastic discount factor**.
#
# In essence, allowing discounting to become a random variable gives us the
# flexibility to combine temporal discounting and attitudes to risk.
#
# We leave further discussion to [other lectures](https://python.quantecon.org/markov_asset.html) 
# because our aim is to move to the computational problem.

# ## Pricing a cash flow
#
# Now let's try to price an asset like a share, which delivers a cash flow $D_t,
# D_{t+1}, \ldots$.
#
# We will call these payoffs "dividends".
#
# If we buy the share, hold it for one period and sell it again, we receive one
# dividend and our payoff is $D_{t+1} + P_{t+1}$.
#
# Therefore, by [](eq:nrnp), the price should be
#
# $$
#     P_t = \mathbb E_t M_{t+1} [ D_{t+1} + P_{t+1} ]
# $$ (lteeqs0)
#
# Because prices generally grow over time, which complicates analysis, it will be
# easier for us to solve for the **price-dividend ratio** $V_t := P_t / D_t$.
#
# Let's write down an expression that this ratio should satisfy.
#
# We can divide both sides of {eq}`lteeqs0` by $D_t$ to get
#
# ```{math}
# :label: pdex
#
# V_t = {\mathbb E}_t \left[ M_{t+1} \frac{D_{t+1}}{D_t} (1 + V_{t+1}) \right]
# ```
#
# We can also write this as
#
# ```{math}
# :label: pdex2
#
# V_t = {\mathbb E}_t \left[ M_{t+1} \exp(G^d_{t+1}) (1 + V_{t+1}) \right]
# ```
#
# $$
#     G^d_{t+1} = \ln \frac{D_{t+1}}{D_t}
# $$
#
# is the growth rate of dividends.
#
# Our aim is to solve [](pdex2) but before that we need to specify
#
# 1. the stochastic discount factor $M_{t+1}$ and
# 1. the growth rate of dividends $G^d_{t+1}$

# ## Choosing the stochastic discount factor
#
# We will adopt the stochastic discount factor described in {cite}`Lucas1978`, which has the form
#
# ```{math}
# :label: lucsdf
#     M_{t+1} = \beta \frac{u'(C_{t+1})}{u'(C_t)}
# ```
#
# where $u$ is a utility function and $C_t$ is time $t$ consumption of a representative consumer.
#
# (An explanation of the ideas behind this expression is given in [a later lecture](https://python-advanced.quantecon.org/lucas_model.html) and we omit further details and motivation.)
#
# For utility, we'll assume the **constant relative risk aversion** (CRRA) specification
#
# ```{math}
# :label: eqCRRA
#     u(c) = \frac{c^{1-\gamma}}{1 - \gamma} 
# ```
#
# Inserting the CRRA specification into {eq}`lucsdf` and letting
#
# $$
#     G^c_{t+1} = \ln \left( \frac{C_{t+1}}{C_t} \right)
# $$ 
#
# the growth rate rate of consumption, we obtain 
#
# ```{math}
# :label: lucsdf2
#     M_{t+1}
#     = \beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}
#     = \beta \exp( G^c_{t+1} )^{-\gamma} 
#     = \beta \exp(-\gamma G^c_{t+1})
# ```

# ## Solving for the price-dividend ratio
#
# Substituting [](lucsdf2) into {eq}`pdex2` gives the price-dividend ratio
# formula
#
# $$
#     V_t = \beta {\mathbb E}_t 
#     \left[ \exp(G^d_{t+1} - \gamma G^c_{t+1}) (1 + V_{t+1}) \right]
# $$ (pdex3)
#
# For now we assume that there is a Markov chain $\{X_t\}$, which we call
# the **state process**,  such that
#
# $$
# \begin{aligned}
#     & G^c_{t+1} = \mu_c + X_t + \sigma_c \epsilon_{c, t+1} \\
#     & G^d_{t+1} = \mu_d + X_t + \sigma_d \epsilon_{d, t+1} 
# \end{aligned}
# $$
#
# Here $\{\epsilon_{c, t}\}$ and $\{\epsilon_{d, t}\}$ are IID and standard
# normal, and independent of eachother.
#
# We can think of $\{X_t\}$ as an aggregate shock that affects both consumption
# growth and firm profits (and hence dividends).
#
# We let $P$ be the [stochastic matrix that governs $\{X_t\}$](https://python.quantecon.org/finite_markov.html) and assume $\{X_t\}$ takes values in some finite set $S$.
#
# We guess that $V_t$ is a function of this state process (and this guess turns
# out to be correct).
#
# This means that $V_t = v(X_t)$ for some unknown function $v$.
#
# By [](pdex3), the unknown function $v$ satisfies the equation
#
# $$
#     v(X_t) = \beta {\mathbb E}_t 
#     \left\{
#         \exp[
#             a + (1-\gamma) X_t + 
#                 \sigma_c \epsilon_{c, t+1} - 
#                 \gamma  \sigma_d \epsilon_{d, t+1}     
#             ]
#         (1 + v(X_{t+1}))
#     \right\}
# $$ (eq:neweqn101)
#
# where $a := \mu_c - \gamma \mu_d$
#
# Since the shocks $\epsilon_{c, t+1}$ and $\epsilon_{d, t+1}$ are independent of
# $\{X_t\}$, we can integrate them out.
#
# We use the following property of lognormal distributions: if $Y = \exp(c
# \epsilon)$ for constant $c$ and $\epsilon \sim N(0,1)$, then $\mathbb E Y =
# \exp(c^2/2)$.
#
# This yields
#
# $$
#     v(X_t) = \beta {\mathbb E}_t 
#     \left\{
#         \exp \left[
#             a + (1-\gamma) X_t + 
#                 \frac{\sigma_c^2 + \gamma^2  \sigma_d^2}{2}
#             \right]
#         (1 + v(X_{t+1}))
#     \right\}
# $$ (eq:ntev)
#
# Conditioning on $X_t = x$, we can write this as
#
# $$
#     v(x) = \beta \sum_{y \in S}
#     \left\{
#         \exp \left[
#             a + (1-\gamma) x + 
#                 \frac{\sigma_c^2 + \gamma^2  \sigma_d^2}{2} 
#             \right]
#         (1 + v(y))
#     \right\}
#     P(x, y)
# $$ (eq:ntecx)
#
# for all $x \in S$.
#
# Suppose $S = \{x_1, \ldots, x_N\}$.
#
# Then we can think of $v$ as an $N$-vector and write
#
# $$
#     v[i] = \beta \sum_{j=1}^N
#     \left\{
#         \exp \left[
#             a + (1-\gamma) x[i] + 
#                 \frac{\sigma_c^2 + \gamma^2  \sigma_d^2}{2} 
#             \right]
#         (1 + v[j])
#     \right\}
#     P[i, j]
# $$ (eq:ntecx2)
#
# for $i = 1, \ldots, N$.
#
# We can write [](eq:ntecx2) in vector form as
#
# $$
#     v = K (\mathbb 1 + v)
# $$ (eq:ntecxv)
#
# where $K$ is the matrix defined by 
#
# $$
#     K[i, j]
#     = \beta \left\{
#         \exp \left[
#             a + (1-\gamma) x[i] + 
#                 \frac{\sigma_c^2 + \gamma^2  \sigma_d^2}{2}
#             \right]
#     \right\} P[i,j]
# $$
#
#
# Notice that [](eq:ntecxv) can be written as $(I - K)v = K \mathbb 1$.
#
# The Neumann series lemma tells us that $(I - K)$ is invertible and the solution
# is
#
# $$
#     v = (I - K)^{-1} K \mathbb 1
# $$ (eq:ntecxvv)
#
# whenever $r(K)$, the spectral radius of $K$, is strictly less than one.
#
# Once we specify $P$ and all the parameters, we can obtain $K$ and 
# then compute the solution [](eq:ntecxvv).
#
#
# ## Code
#
# We assume that $\{X_t\}$ is a discretization of the AR(1) process
#
# $$
#     X_{t+1} = \rho X_t + \sigma \eta_{t+1}
# $$
#
# where $\rho, \sigma$ are parameters and $\{\eta_t\}$ is IID and standard normal.
#
# To discretize this process we use QuantEcon.py's `tauchen` function.

# +
Model = namedtuple('Model', 
                   ('P', 'S', 'β', 'γ', 'μ_c', 'μ_d', 'σ_c', 'σ_d'))

def create_model(N=100,         # size of state space for Markov chain
                 ρ=0.2,         # persistence parameter for Markov chain
                 σ=0.1,         # persistence parameter for Markov chain
                 β=0.98,        # discount factor
                 γ=2.5,         # coefficient of risk aversion 
                 μ_c=0.01,      # mean growth of consumtion
                 μ_d=0.01,      # mean growth of dividends
                 σ_c=0.02,      # consumption volatility 
                 σ_d=0.04):     # dividend volatility 

    mc = qe.tauchen(N, ρ, σ, 0)
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
    N = len(S)
    # Reshape and multiply pointwise using broadcasting
    x = np.reshape(S, (N, 1))
    a = μ_c - γ * μ_d
    e = np.exp(a + (1 - γ) * x + (σ_c**2 + γ**2 * σ_d**2) / 2)
    K = β * e * P
    return K

def compute_K_loop(model):
    # Setp up
    P, S, β, γ, μ_c, μ_d, σ_c, σ_d = model
    N = len(S)
    K = np.empty((N, N))
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
    N = len(model.S)
    # Make sure that a unique solution exists
    test_stability(K)

    # Compute v
    I = np.identity(N)
    Ones = np.ones(N)
    v = np.linalg.solve(I - K, K @ Ones)

    return v


# -

# Here's a plot of $v$ as a function of the state for several values of $\gamma$,
# with a positively correlated Markov process and $g(x) = \exp(x)$

# +
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
# -

# Notice that $v$ is decreasing in each case.
#
# This is because, with a positively correlated state process, higher states indicate higher future consumption growth.
#
# With the stochastic discount factor {eq}`lucsdf2`, higher growth decreases the
# discount factor, lowering the weight placed on future dividends.
#
#
# ## An Extended Example
#
# One problem with the last set is that volatility is constant through time (i.e.,
# $\sigma_c$ and $\sigma_d$ are constants).
#
# In reality, financial markets and growth rates of macroeconomic variables
# exhibit bursts of volatility.
#
# To accommodate this, we now suppose that
#
# $$
# \begin{aligned}
#     & G^c_{t+1} = \mu_c + Z_t + \exp(H^c_t) \epsilon_{c, t+1} \\
#     & G^d_{t+1} = \mu_d + Z_t + \exp(H^d_t) \epsilon_{d, t+1} 
# \end{aligned}
# $$
#
# where $\{Z_t\}$ is a finite Markov chain and $\{H^c_t\}$ and $\{H^d_t\}$ are AR(1) processes of the form
#
# $$
# \begin{aligned}
#     H^c_{t+1} & = \rho_c H^c_t + \sigma_c \eta_{c, t+1}  \\
#     H^d_{t+1} & = \rho_d H^d_t + \sigma_d \eta_{d, t+1}  
# \end{aligned}
# $$
#
# Here $\{\eta^c_t\}$ and $\{\eta^d_t\}$ are IID and standard normal.
#
# Let $X_t = (H^c_t, H^d_t, Z_t)$.
#
# We call $\{X_t\}$ the state process and guess that $V_t$ is a function of
# this state process, so that $V_t = v(X_t)$ for some unknown function $v$.
#
# Modifying [](eq:neweqn101) to accommodate the new growth specifications, 
# we find that $v$ satisfies 
#
# $$
#     v(X_t) = \beta {\mathbb E}_t 
#     \left\{
#         \exp[
#             a + (1-\gamma) Z_t + 
#                 \exp(H^c_t) \epsilon_{c, t+1} - 
#                 \gamma \exp(H^d_t) \epsilon_{d, t+1}     
#             ]
#         (1 + v(X_{t+1}))
#     \right\}
# $$ (eq:neweqn102)
#
# Conditioning on state $x = (h_c, h_d, z)$, this becomes
#
# $$
#     v(x) = \beta  {\mathbb E}_t
#         \exp[
#             a + (1-\gamma) z + 
#                 \exp(h_c) \epsilon_{c, t+1} - 
#                 \gamma \exp(h_d) \epsilon_{d, t+1}     
#             ]
#         (1 + v(X_{t+1}))
# $$ (eq:neweqn103)
#
# As before, we integrate out the independent shocks and use the rules for 
# expectations of lognormals to obtain
#
# $$
#     v(x) = \beta  {\mathbb E}_t
#         \exp \left[
#             a + (1-\gamma) z + 
#                 \frac{\exp(2 h_c) + \gamma^2 \exp(2 h_d)}{2}
#             \right]
#         (1 + v(X_{t+1}))
# $$ (eq:neweqn103)
#
# Using the definition of the state and setting
#
# $$
#     \kappa(h_c, h_z, z) :=
#         \exp \left[
#             a + (1-\gamma) z + 
#                 \frac{\exp(2 h_c) + \gamma^2 \exp(2 h_d)}{2}
#             \right]
# $$
#
# we can write this more explicitly
#
# $$
#     v(h_c, h_d, z) = 
#     \beta \sum_{h_c', h_d', z'}
#         \kappa(h_c, h_z, z)
#         (1 + v(h_c', h_d', z')) P(h_c, h_c')Q(h_d, h_d')R(z, z')
# $$ (eq:neweqn104)
#
# Here $P, Q, R$ are the stochastic matrices for, respectively, discretized
# $\{H^c_t\}$, discretized $\{H^d_t\}$ and $\{Z_t\}$.
#
# Let's now write the state using indices, with $(i, j, k)$ being the
# indices for $(h_c, h_z, z)$.
#
# The last expression becomes
#
# $$
#     v[i, j, k] = 
#     \beta \sum_{i', j', k'}
#         \kappa[i, j, k]
#         (1 + v[i', j', k']) P[i, i']Q[j, j']R[k, k']
# $$ (eq:neweqn104)
#
# If we define the linear operator
#
# $$
#     (Kg)[i, j, k] = 
#     \beta \sum_{i', j', k'}
#         \kappa[i, j, k] g[i', j', k'] P[i, i']Q[j, j']R[k, k']
# $$
#
# then [](eq:neweqn104) becomes $v(i, j, k) = (K(1 + v))(i, j, k)$, or, in vector
# form,
#
# $$
#     v = K(\mathbb 1 + v)
# $$
#
# Provided that $K$ is invertible, the solution is given by
#
# $$
#     v = (I - K)^{-1} K \mathbb 1
# $$


Model = namedtuple('Model', 
                   ('P', 'S', 'β', 'γ', 'μ_c', 'μ_d', 'σ_c', 'σ_d'))

def create_model(N=100,         # size of state space for Markov chain
                 ρ=0.2,         # persistence parameter for Markov chain
                 σ=0.1,         # persistence parameter for Markov chain
                 β=0.98,        # discount factor
                 γ=2.5,         # coefficient of risk aversion 
                 μ_c=0.01,      # mean growth of consumtion
                 μ_d=0.01,      # mean growth of dividends
                 σ_c=0.02,      # consumption volatility 
                 σ_d=0.04):     # dividend volatility 

    mc = qe.tauchen(N, ρ, σ, 0)
    S = mc.state_values
    P = mc.P
    return Model(P=P, S=S, β=β, γ=γ, μ_c=μ_c, μ_d=μ_d, σ_c=σ_c, σ_d=σ_d)

