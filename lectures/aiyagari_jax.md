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

+++ {"colab_type": "text", "id": "view-in-github"}

<a href="https://colab.research.google.com/gist/shlff/5bde3e216c5f65dac9763eb17c549044/aiyagari_jax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++ {"id": "b205a26b"}

# The Aiyagari Model

```{code-cell}
:id: 46444d67
:incorrectly_encoded_metadata: hide-output=false

#!pip install --upgrade quantecon  #  Uncomment if necessary
```

+++ {"id": "67cab042"}

We use the following imports

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: FZrKQ6DLVXjI
outputId: a012dba5-9ec6-4854-955a-f0ccf8522b5d
---
!nvidia-smi
```

```{code-cell}
:id: 5f04cbeb
:incorrectly_encoded_metadata: hide-output=false

import matplotlib.pyplot as plt
import numpy as np
# import quantecon as qe
import jax
import jax.numpy as jnp
```

+++ {"id": "0da8a153"}

We use 64 bit floats with JAX in order to match NumPy/Numba code.

```{code-cell}
:id: 32f6b2f7

jax.config.update("jax_enable_x64", True)
```

+++ {"id": "70499004"}

We will use the following function to compute stationary distributions of stochastic matrices.  (For a reference to the algorithm, see p. 88 of [Economic Dynamics](https://johnstachurski.net/edtc).)

```{code-cell}
:id: a33b72dc

@jax.jit
def compute_stationary(P):
    "Compute the stationary distribution of P by matrix inversion."
    n = len(P)
    I = jnp.identity(n)
    O = jnp.ones((n, n))
    A = I - jnp.transpose(P) + O
    return jnp.linalg.solve(A, jnp.ones(n))
```

+++ {"id": "7cb8283d"}

## Overview

In this lecture, we describe the structure of a class of models that build on work by Truman Bewley [[Bew77](https://python.quantecon.org/zreferences.html#id173)].

We begin by discussing an example of a Bewley model due to Rao Aiyagari [[Aiy94](https://python.quantecon.org/zreferences.html#id137)].

The model features

- Heterogeneous agents  
- A single exogenous vehicle for borrowing and lending  
- Limits on amounts individual agents may borrow  


The Aiyagari model has been used to investigate many topics, including

- precautionary savings and the effect of liquidity constraints [[Aiy94](https://python.quantecon.org/zreferences.html#id137)]  
- risk sharing and asset pricing [[HL96](https://python.quantecon.org/zreferences.html#id129)]  
- the shape of the wealth distribution [[BBZ15](https://python.quantecon.org/zreferences.html#id130)]

+++ {"id": "ce236bad"}

### References

The primary reference for this lecture is [[Aiy94](https://python.quantecon.org/zreferences.html#id137)].

A textbook treatment is available in chapter 18 of [[LS18](https://python.quantecon.org/zreferences.html#id182)].

A continuous time version of the model by SeHyoun Ahn and Benjamin Moll can be found [here](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/aiyagari_continuous_time.ipynb).

+++ {"id": "056a1294"}

## Firms

Firms produce output by hiring capital and labor.

Firms act competitively and face constant returns to scale.

Since returns to scale are constant the number of firms does not matter.

Hence we can consider a single (but nonetheless competitive) representative firm.

The firm’s output is

$$
Y_t = A K_t^{\alpha} N^{1 - \alpha}
$$

where

- $ A $ and $ \alpha $ are parameters with $ A > 0 $ and $ \alpha \in (0, 1) $  
- $ K_t $ is aggregate capital  
- $ N $ is total labor supply (which is constant in this simple version of the model)  


The firm’s problem is

$$
\max_{K, N} \left\{ A K_t^{\alpha} N^{1 - \alpha} - (r + \delta) K - w N \right\}
$$

The parameter $ \delta $ is the depreciation rate.

From the first-order condition with respect to capital, the firm’s inverse demand for capital is


<a id='equation-aiy-rgk'></a>
$$
r = A \alpha  \left( \frac{N}{K} \right)^{1 - \alpha} - \delta \tag{70.1}
$$

Using this expression and the firm’s first-order condition for labor, we can pin down
the equilibrium wage rate as a function of $ r $ as


<a id='equation-aiy-wgr'></a>
$$
w(r) = A  (1 - \alpha)  (A \alpha / (r + \delta))^{\alpha / (1 - \alpha)} \tag{70.2}
$$

+++ {"id": "f631594c"}

These parameters and equations are stored in the following class.

```{code-cell}
:id: ec209518

class Firm:
    
    def __init__(self, 
                 A=1.0,
                 N=1.0,
                 α=0.33,
                 β=0.96,
                 δ=0.05):
        
        self.A, self.N, self.α, self.β, self.δ = A, N, α, β, δ

    def rd(self, K):
        """
        Inverse demand curve for capital.  The interest rate associated with a
        given demand for capital K.
        """
        A, N, α, β, δ = self.A, self.N, self.α, self.β, self.δ
        return A * α * (N / K)**(1 - α) - δ
    
    
    def r_to_w(self, r):
        """
        Equilibrium wages associated with a given interest rate r.
        """
        A, N, α, β, δ = self.A, self.N, self.α, self.β, self.δ
        return A * (1 - α) * (A * α / (r + δ))**(α / (1 - α))
```

+++ {"id": "de716fe2"}

## Households

Infinitely lived households / consumers face idiosyncratic income shocks.

A unit interval of  *ex-ante* identical households face a common borrowing constraint.

The savings problem faced by a typical  household is

$$
\max \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

subject to

$$
a_{t+1} + c_t \leq w z_t + (1 + r) a_t
\quad
c_t \geq 0,
\quad \text{and} \quad
a_t \geq -B
$$

where

- $ c_t $ is current consumption  
- $ a_t $ is assets  
- $ z_t $ is an exogenous component of labor income capturing stochastic unemployment risk, etc.  
- $ w $ is a wage rate  
- $ r $ is a net interest rate  
- $ B $ is the maximum amount that the agent is allowed to borrow  


The exogenous process $ \{z_t\} $ follows a finite state Markov chain with given stochastic matrix $ P $.


In this simple version of the model, households supply labor  inelastically because they do not value leisure.

Below we provide code to solve the household problem,taking $r$ and $w$ as fixed.

+++ {"id": "feb12589"}

### Primitives and Operators

+++ {"id": "0228f274"}

This class stores the parameters that define a household asset 
accumulation problem and the grids used to solve it.

```{code-cell}
:id: aafea72f
:incorrectly_encoded_metadata: hide-output=false

class Household:

    def __init__(self,
                r=0.01,                      # Interest rate
                w=1.0,                       # Wages
                β=0.96,                      # Discount factor
                Π=[[0.9, 0.1], [0.1, 0.9]],  # Markov chain
                z_grid=[0.1, 1.0],           # Exogenous states
                a_min=1e-10, a_max=20,       # Asset grid
                a_size=200):

        # Store values, set up grids over a and z
        self.r, self.w, self.β = r, w, β
        self.a_size = a_size
        self.a_grid = jnp.linspace(a_min, a_max, a_size)
        z_grid, Π = map(np.array, (z_grid, Π))
        self.Π = jax.device_put(Π)
        self.z_grid = jax.device_put(z_grid)
        self.z_size = len(z_grid)
        
    def constants(self):
        return self.r, self.w, self.β
    
    def sizes(self):
        return self.a_size, self.z_size
    
    def arrays(self):
        return self.a_grid, self.z_grid, self.Π
```

+++ {"id": "a7542c38"}

This is the right-hand side of the Bellman equation for the household:

```{code-cell}
:id: e82d1d72

def B(v, constants, sizes, arrays):
    """
    A vectorized version of the right-hand side of the Bellman equation 
    (before maximization), which is a 3D array representing

        B(a, z, a') = u(wz + (1+r)a - a') + β Σ_z' v(a', z') Π(z, z')

    for all (a, z, a').
    """

    # Unpack 
    r, w, β = constants
    a_size, z_size = sizes
    a_grid, z_grid, Π = arrays

    # Compute current consumption as array c[i, j, ip]
    a  = jnp.reshape(a_grid, (a_size, 1, 1))    # a[i]   ->  a[i, j, ip]
    z  = jnp.reshape(z_grid, (1, z_size, 1))    # z[j]   ->  z[i, j, ip]
    ap = jnp.reshape(a_grid, (1, 1, a_size))    # ap[ip] -> ap[i, j, ip]
    c = w*z + (1 + r)*a - ap

    # Calculate continuation rewards at all combinations of (a, z, ap)
    v = jnp.reshape(v, (1, 1, a_size, z_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Π = jnp.reshape(Π, (1, z_size, 1, z_size))  # Π[j, jp]  -> Π[i, j, ip, jp]
    EV = jnp.sum(v * Π, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return jnp.where(c > 0, jnp.log(c) + β * EV, -jnp.inf)
```

+++ {"id": "67ee981b"}

The next function computes greedy policies.

```{code-cell}
:id: 980ffb29

def get_greedy(v, constants, sizes, arrays):
    "Computes a v-greedy policy, returned as a set of indices."
    return jnp.argmax(B(v, constants, sizes, arrays), axis=2)
```

+++ {"id": "3c1801df"}

We need to know rewards at a given policy for policy iteration.

```{code-cell}
:id: f5852bf4

def compute_r_σ(σ, constants, sizes, arrays):
    """
    Compute the array r_σ[i, j] = r[i, j, σ[i, j]], which gives current
    rewards given policy σ.
    """

    # Unpack
    r, w, β = constants
    a_size, z_size = sizes
    a_grid, z_grid, Π = arrays

    # Compute r_σ[i, j]
    a = jnp.reshape(a_grid, (a_size, 1))
    z = jnp.reshape(z_grid, (1, z_size))
    ap = a_grid[σ]
    c = (1 + r)*a + w*z - ap
    r_σ = jnp.log(c)

    return r_σ
```

+++ {"id": "9779acee"}

The following linear operator is also needed for policy iteration.

```{code-cell}
:id: ee6f3c05

def R_σ(v, σ, constants, sizes, arrays):
    """
    The value v_σ of a policy σ is defined as 

        v_σ = (I - β P_σ)^{-1} r_σ

    Here we set up the linear map v -> R_σ v, where R_σ := I - β P_σ. 

    In the consumption problem, this map can be expressed as

        (R_σ v)(a, z) = v(a, z) - β Σ_z' v(σ(a, z), z') Π(z, z')

    Defining the map as above works in a more intuitive multi-index setting
    (e.g. working with v[i, j] rather than flattening v to a one-dimensional
    array) and avoids instantiating the large matrix P_σ.

    """
    # Unpack
    r, w, β = constants
    a_size, z_size = sizes
    a_grid, z_grid, Π = arrays

    # Set up the array v[σ[i, j], jp]
    zp_idx = jnp.arange(z_size)
    zp_idx = jnp.reshape(zp_idx, (1, 1, z_size))
    σ = jnp.reshape(σ, (a_size, z_size, 1))
    V = v[σ, zp_idx]

    # Expand Π[j, jp] to Π[i, j, jp]
    Π = jnp.reshape(Π, (1, z_size, z_size))

    # Compute and return v[i, j] - β Σ_jp v[σ[i, j], jp] * Π[j, jp]
    return v - β * jnp.sum(V * Π, axis=2)
```

+++ {"id": "d47f25b0"}

The next function computes the lifetime value of a given policy.

```{code-cell}
:id: d9ce4bf5

def get_value(σ, constants, sizes, arrays):
    "Get the value v_σ of policy σ by inverting the linear map R_σ."

    r_σ = compute_r_σ(σ, constants, sizes, arrays)
    # Reduce R_σ to a function in v
    partial_R_σ = lambda v: R_σ(v, σ, constants, sizes, arrays)
    # Compute inverse v_σ = (I - β P_σ)^{-1} r_σ
    return jax.scipy.sparse.linalg.bicgstab(partial_R_σ, r_σ)[0]
```

+++ {"id": "qiRnd8M57LZ1"}

The following function is used for optimistic policy iteration.

```{code-cell}
:id: fKJFyp1cE5fk

def T_σ(v, σ, constants, sizes, arrays):
    "The σ-policy operator."

    # Unpack model
    γ, w, β = constants
    a_size, z_size = sizes
    a_grid, z_grid, Π = arrays

    r_σ = compute_r_σ(σ, constants, sizes, arrays)

    # Compute the array v[σ[i, j], jp]
    zp_idx = jnp.arange(z_size)
    zp_idx = jnp.reshape(zp_idx, (1, 1, z_size))
    σ = jnp.reshape(σ, (a_size, z_size, 1))
    V = v[σ, zp_idx]      

    # Convert Q[j, jp] to Q[i, j, jp] 
    Π = jnp.reshape(Π, (1, z_size, z_size))

    # Calculate the expected sum Σ_jp v[σ[i, j], jp] * Q[i, j, jp]
    Ev = np.sum(V * Π, axis=2)

    return r_σ + β * np.sum(V * Π, axis=2)
```

+++ {"id": "5a53ac60"}

Let's target these functions for JIT-compilation.

```{code-cell}
:id: 63fedaa0

B = jax.jit(B, static_argnums=(2,))
compute_r_σ = jax.jit(compute_r_σ, static_argnums=(2,))
get_greedy = jax.jit(get_greedy, static_argnums=(2,))
get_value = jax.jit(get_value, static_argnums=(2,))
R_σ = jax.jit(R_σ, static_argnums=(3,))
T_σ = jax.jit(T_σ, static_argnums=(3,))
```

+++ {"id": "82f0c26b"}

## Solvers

We will solve the household problem using Howard policy iteration.

```{code-cell}
:id: 24ec7591

def policy_iteration(household, verbose=True):
    "Howard policy iteration routine."
    constants = household.constants()
    sizes = household.sizes()
    arrays = household.arrays()
    
    vz = jnp.zeros(sizes)
    σ = jnp.zeros(sizes, dtype=int)
    i, error = 0, 1.0
    while error > 0:
        v_σ = get_value(σ, constants, sizes, arrays)
        σ_new = get_greedy(v_σ, constants, sizes, arrays)
        error = jnp.max(jnp.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        if verbose:
            print(f"Concluded loop {i} with error {error}.")
    return σ
```

+++ {"id": "ESPsmqkOC7X-"}

We can also solve the problem using optimistic policy iteration.

```{code-cell}
:id: oxT7nm6-Ec7O

def optimistic_policy_iteration(household, tol=1e-5, m=10):
      constants = household.constants()
      sizes = household.sizes()
      arrays = household.arrays()

      v = jnp.zeros(sizes)
      error = tol + 1
      while error > tol:
        last_v = v
        σ = get_greedy(v, constants, sizes, arrays)
        for _ in range(m):
          v = T_σ(v, σ, constants, sizes, arrays)
        error = jnp.max(np.abs(v - last_v))
      return get_greedy(v, constants, sizes, arrays) 
```

+++ {"id": "32c0fa14"}

As a first example of what we can do, let’s compute and plot an optimal accumulation policy at fixed prices.

```{code-cell}
:id: 880d56db
:incorrectly_encoded_metadata: hide-output=false

# Example prices
r = 0.03
w = 0.956

# Create an instance of Housbehold
household = Household(r=r, w=w)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: mM1eRTmPY5xY
outputId: e465e724-fe6c-44bb-f435-613df9efc87c
---
%%time 
σ_star_hpi = policy_iteration(household)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 718da8af
outputId: 02e76f03-4dec-4739-e5ae-5d70a6a6f58b
---
%%time 
σ_star = optimistic_policy_iteration(household)
```

+++ {"id": "077dcf0d"}

The next plot shows asset accumulation policies at different values of the exogenous state.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 773
id: '63942264'
incorrectly_encoded_metadata: hide-output=false
outputId: 1dc7dc61-2e84-4e30-a940-6c1033c5b8e7
---
a_size, z_size = household.sizes()
a_grid, z_grid, Π = household.arrays()

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(a_grid, a_grid, 'k--')  # 45 degrees
for j in range(z_size):
    lb = f'$z = {z_grid[j]:.2}$'
    ax.plot(a_grid, a_grid[σ_star[:, j]], lw=2, alpha=0.6, label=lb)
    ax.set_xlabel('current assets')
    ax.set_ylabel('next period assets')
ax.legend(loc='upper left')

plt.show()
```

+++ {"id": "07f90445"}

### Capital Supply

To start thinking about equilibrium, we need to know how much capital households supply at a given interest rate $r$.

This quantity can be calculated by taking the stationary distribution of assets under the optimal policy and computing the mean.

The next function implements this calculation for a given policy $\sigma$.

```{code-cell}
:id: 037b3e6b

def compute_asset_stationary(σ, constants, sizes, arrays):
    """
    Compute the stationary distribution of assets under the policy σ.
    
    First we compute the stationary distribution of P_σ, which is for the 
    bivariate Markov chain of the state (a_t, z_t).  Then we sum out
    z_t to get the marginal distribution for a_t.
    
    """
    
    # Unpack
    r, w, β = constants
    a_size, z_size = sizes
    a_grid, z_grid, Π = arrays
    
    # Construct P_σ as an array of the form P_σ[i, j, ip, jp]
    ap_idx = jnp.arange(a_size)
    ap_idx = jnp.reshape(ap_idx, (1, 1, a_size, 1))
    σ = jnp.reshape(σ, (a_size, z_size, 1, 1))
    A = jnp.where(σ == ap_idx, 1, 0)
    Π = jnp.reshape(Π, (1, z_size, 1, z_size))
    P_σ = A * Π
    
    # Reshape P_σ into a matrix
    n = a_size * z_size
    P_σ = jnp.reshape(P_σ, (n, n))
    
    # Get stationary distribution and reshape onto [i, j] grid
    ψ = compute_stationary(P_σ)
    ψ = jnp.reshape(ψ, (a_size, z_size))
    
    # Sum along the rows to get the marginal distribution of assets
    ψ_a = jnp.sum(ψ, axis=1)
    return ψ_a
```

+++ {"id": "467965cb"}

We will also target this function for JIT compilation.

```{code-cell}
:id: d208ec9a

compute_asset_stationary = jax.jit(compute_asset_stationary, 
                                  static_argnums=(2,))
```

+++ {"id": "2b2e921e"}

Let's give this a test run.

```{code-cell}
:id: 336ffaeb

constants = household.constants()
sizes = household.sizes()
arrays = household.arrays()
ψ = compute_asset_stationary(σ_star, constants, sizes, arrays)
```

+++ {"id": "7ffddabf"}

The distribution should sum to one:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: ef0e2c95
outputId: 2b6ed962-c13c-4ec8-b95d-d8de66b6f581
---
ψ.sum()
```

+++ {"id": "19e8915a"}

Now we are ready to compute capital supply by households given wages and interest rates.

```{code-cell}
:id: 5ad39472
:incorrectly_encoded_metadata: hide-output=false

def capital_supply(household):
    """
    Map household decisions to the induced level of capital stock.
    """
    constants = household.constants()
    sizes = household.sizes()
    arrays = household.arrays()
    
    # Compute the optimal policy
    σ_star = optimistic_policy_iteration(household)
    # Compute the stationary distribution
    ψ_a = compute_asset_stationary(σ_star, constants, sizes, arrays)
    # Return K
    return float(jnp.sum(ψ_a * household.a_grid))
```

+++ {"id": "753a857a"}

## Equilibrium

We construct  a *stationary rational expectations equilibrium* (SREE).

In such an equilibrium

- prices induce behavior that generates aggregate quantities consistent with the prices  
- aggregate quantities and prices are constant over time  


In more detail, an SREE lists a set of prices, savings and production policies such that

- households want to choose the specified savings policies taking the prices as given  
- firms maximize profits taking the same prices as given  
- the resulting aggregate quantities are consistent with the prices; in particular, the demand for capital equals the supply  
- aggregate quantities (defined as cross-sectional averages) are constant  


In practice, once parameter values are set, we can check for an SREE by the following steps

1. pick a proposed quantity $ K $ for aggregate capital  
1. determine corresponding prices, with interest rate $ r $ determined by [(70.1)](#equation-aiy-rgk) and a wage rate $ w(r) $ as given in [(70.2)](#equation-aiy-wgr)  
1. determine the common optimal savings policy of the households given these prices  
1. compute aggregate capital as the mean of steady state capital given this savings policy  


If this final quantity agrees with $ K $ then we have a SREE.  Otherwise we adjust $K$.

These steps describe a fixed point problem which we solve below.

+++ {"id": "315fdbbe"}

Let’s inspect visually as a first pass.

The following code draws aggregate supply and demand curves for capital.

The intersection gives equilibrium interest rates and capital.

```{code-cell}
:id: 821ed7ad
:incorrectly_encoded_metadata: hide-output=false

# Create default instances
household = Household()
firm = Firm()

# Create a grid of r values at which to compute demand and supply of capital
num_points = 50
r_vals = np.linspace(0.005, 0.04, num_points)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 8fI4gTHlyqaE
outputId: 55bb5732-3451-43db-d1d0-6d997303b088
---
r_vals
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 86f080d1
incorrectly_encoded_metadata: hide-output=false
outputId: 9a902730-91a5-41d6-8b2f-95e9c158f0d5
---
%%time
# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    household.r = r
    household.w = firm.r_to_w(r)
    k_vals[i] = capital_supply(household)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 449
id: 72c8f573
incorrectly_encoded_metadata: hide-output=false
outputId: cb745333-0325-4d74-9f91-dad34dd03ccd
---
# Plot against demand for capital by firms
fig, ax = plt.subplots()
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital')
ax.plot(k_vals, firm.rd(k_vals), lw=2, alpha=0.6, label='demand for capital')
ax.grid()
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.legend(loc='upper right')

plt.show()

# Solve for the fixed point
```

```{code-cell}
:id: a2a41546

def excess_demand(K):
    r = firm.rd(K)
    w = firm.r_to_w(r)
    household.r, household.w = r, w
    return K - capital_supply(household)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: KqhJ9O7WWZPj
outputId: ff47f31c-bb2e-4a16-8b5a-647fa3cd9c5c
---
k = 8.5
excess_demand(k)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: NSGtQ--_f04M
outputId: b03e170d-0eef-486e-a3dc-7978969ed6e9
---
%%time
num_points = 50
k_vals = np.linspace(4, 12, num_points)
out = [excess_demand(k) for k in k_vals]
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 449
id: a2e06bba
incorrectly_encoded_metadata: hide-output=false
outputId: dd00ff98-0b7d-4432-d400-5e78abafd299
---
fig, ax = plt.subplots()
ax.plot(k_vals, out, lw=2, alpha=0.6, label='excess demand')
ax.plot(k_vals, np.zeros_like(k_vals), 'k--', label="45")
ax.set_xlabel('capital')
ax.legend()
plt.show()
```

```{code-cell}
:id: 6856ca46

def bisect(f, a, b, tol=10e-2):
    """
    Implements the bisection root finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b

    while upper - lower > tol:
        middle = 0.5 * (upper + lower)
        if f(middle) > 0:   # root is between lower and middle
            lower, upper = lower, middle
        else:               # root is between middle and upper
            lower, upper = middle, upper

    return 0.5 * (upper + lower)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: f539dade
outputId: 3dcd97a1-d437-4415-9733-73e5f3657826
---
bisect(excess_demand, 6.0, 10.0)
```

```{code-cell}
:id: 249e9fd5

def compute_equilibrium(household, firm):

    def excess_supply(K):
        r = firm.rd(K)
        w = firm.r_to_w(r)
        household.r, household.w = r, w
        return capital_supply(household) - K

    solution = bisect(excess_demand, 6.0, 10.0)
    return solution
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: e5db8933
outputId: cf3392c7-7abe-4068-e09d-b8874fd98d6a
---
%%time
household = Household()
firm = Firm()
compute_equilibrium(household, firm)
```

+++ {"id": "8o_oevnQY_3q"}

# Simulations

```{code-cell}
:id: 6099487a


```

```{code-cell}
:id: 7xxReHEiTJqV


```

```{code-cell}
:id: iROq1aa1ZaSU


```
