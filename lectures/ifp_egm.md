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

# Income Fluctuations and Endogenous Grid Method

```{include} _admonition/gpu.md
```

```{code-cell} ipython3
:tags: [hide-output]
!pip install --upgrade quantecon interpolation
```

```{code-cell} ipython3
import quantecon as qe
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from interpolation import interp
from numba import njit, float64
from numba.experimental import jitclass
```

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

## Setup of the problem


Consider a household that chooses a state-contingent consumption plan $\{c_t\}_{t \geq 0}$ to maximize

$$
\mathbb{E} \, \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

subject to

$$
a_{t+1} \leq  R(a_t - c_t)  + Y_{t+1},
\quad c_t \geq 0,
\quad a_t \geq 0
\quad t = 0, 1, \ldots
$$

and CRRA utility

$$
u(c) = \frac{c^{1 - \gamma}} {1 - \gamma}
$$

Reference: https://python.quantecon.org/ifp.html#the-optimal-savings-problem

```{code-cell} ipython3
def ifp(r=0.01,
        β=0.96,
        grid_max=16,
        grid_size=200, 
        ρ=0.99, ν=0.02, y_size=25):
  

    R = 1 + r
    β, R= jax.device_put([β, R])
    
    mc = qe.tauchen(y_size, ρ, ν)
    y_grid, P = jnp.exp(mc.state_values), mc.P
    P, y_grid = jnp.array(P), jnp.array(y_grid)
    
    n = len(P)
    
    grid = jnp.linspace(0, grid_max, grid_size)
    sizes = n, grid_size, y_size
    grid, y_grid, P = jax.device_put([grid, y_grid, P])

    # Recall that we need R β < 1 for convergence.
    assert R * β < 1, "Stability condition violated."
        
    return (β, R), sizes, (grid, y_grid, P)

@jax.jit
def u_prime(c, γ=1.5):
    return c**(-γ)

@jax.jit
def u_prime_inv(u_prime, γ=1.5):
        return u_prime**(-1/γ)
```

Generate an instance.

```{code-cell} ipython3
model = ifp()

constants, sizes, arrays = model
    
β, R = constants
n, grid_size, y_size = sizes
grid, y_grid, P = arrays
```

We can plot a sample of the income process.

```{code-cell} ipython3
length_ts = 1_000  # set the length for simulated income

fig, ax = plt.subplots()
mc_new = qe.MarkovChain(P, y_grid)
y = mc_new.simulate(length_ts)
ax.plot(y)
ax.set_xlabel('time')
ax.set_ylabel('income')
plt.show()
```

## Defining vectorized $K$

### Value function and Euler function

The *value function* $V \colon \mathsf S \to \mathbb{R}$ is defined by

$$
V(a, z) := \max \, \mathbb{E}
\left\{
\sum_{t=0}^{\infty} \beta^t u(c_t)
\right\}
$$

The corresponding Euler equation is

$$
u' (c_t)
= \max \left\{
    \beta R \,  \mathbb{E}_t  u'(c_{t+1})  \,,\;  u'(a_t)
\right\}
$$

Reference: https://python.quantecon.org/ifp.html#value-function-and-euler-equation

+++

### Optimal consumption

To solve the optimization problem is to compute the optimal consumption policy $\sigma^*: S \to \mathbb R$ s.t. 

$$
(a_0, z_0) = (a, z),
\quad
c_t = \sigma^*(a_t, Z_t)
\quad \text{and} \quad
a_{t+1} = R (a_t - c_t) + Y_{t+1}
$$

Reference:
- https://python.quantecon.org/zreferences.html#id253
- https://python.quantecon.org/ifp.html#optimality-results

+++

### IFP with EGM

Rewrite the Euler equation in functional form

$$
(u' \circ \sigma)  (a, z)
= \max \left\{
\beta R \, \mathbb E_z (u' \circ \sigma)
    [R (a - \sigma(a, z)) + \hat Y, \, \hat Z]
\, , \;
     u'(a)
     \right\}
$$


where $(u' \circ \sigma)(s) := u'(\sigma(s))$.

Let $\mathscr C$ be the space of continuous functions $\sigma \colon \mathbf S \to \mathbb R$ such that $\sigma$ is increasing in the first argument, $0 < \sigma(a,z) \leq a$ for all $(a,z) \in \mathbf S$, and

$$
\sup_{(a,z) \in \mathbf S}
\left| (u' \circ \sigma)(a,z) - u'(a) \right| < \infty
$$

For given $\sigma \in \mathscr{C}$, define $K \sigma (a,z)$ as the unique $c \in [0, a]$ that solves

$$
u'(c)
= \max \left\{
           \beta R \, \mathbb E_z (u' \circ \sigma) \,
           [R (a - c) + \hat Y, \, \hat Z]
           \, , \;
           u'(a)
     \right\}
$$

where $K$ is the Coleman--Reffett operator.

Reference here: https://python.quantecon.org/ifp.html#computation

+++

EGM is to take a grid of saving values $s_i=a_i - c_i$.

Also we assume $u$ is invertible.

$$
c
= (u')^{-1}\left (\max \left\{
           \beta R \, \mathbb E_z (u' \circ \sigma) \,
           [R (a - c) + \hat Y, \, \hat Z]
           \, , \;
           u'(a)
     \right\} \right )
$$

+++

On one hand, $a_0=s_0=0$.

On the other hand since $s>0$ implies $c<a$ consumption is interior.

Hence we can drop out the max and we solve for 

$$
c_i =
(u')^{-1}
\left\{
    \beta \, R \mathbb E_z
    (u' \circ \sigma) \, [\hat R s_i + \hat Y, \, \hat Z]
\right\}
$$ (euler)
for each $s_i$.

Reference: https://python.quantecon.org/ifp_advanced.html#using-an-endogenous-grid

+++

### Jax version EGM

First we define a vectorized operator $K$ based on {eq}`euler`.

```{code-cell} ipython3
def K_egm_vec(a_in, σ_in, constants, sizes, arrays):
    """
    The vectorzied operator K using EGM.

    """
    
    # Unpack
    β, R = constants
    n, grid_size, y_size = sizes
    grid, y_grid, P = arrays
    
    # Linearly interpolate σ(a, z)
    def σ(a, z):
        return jnp.interp(a, a_in[:,z], σ_in[:,z])

    σ_vec = jnp.vectorize(σ)

    
    # Broadcast and vectorize
    z_grid = jnp.reshape(jnp.arange(n), (1, n, 1))
    z_hat_grid = jnp.reshape(jnp.arange(n), (1, 1, n))
    s_grid = jnp.reshape(grid, (grid_size, 1, 1))
    
    # Evaluate σ_out
    a_next = R * s_grid + y_grid[z_hat_grid]
    σ_next = σ_vec(a_next, z_hat_grid)
    up = u_prime(σ_next)
    P = jnp.reshape(P, (1, n, n))
    E = jnp.sum(up * P, axis=-1)

    σ_out = u_prime_inv(β * R * E)

    # Compute a_out by s = a - c
    a_out = grid[:, jnp.newaxis] + σ_out
    
    # Set σ_0 = 0 and a_0 = 0
    σ_out = σ_out.at[0, :].set(0)
    a_out = a_out.at[0, :].set(0)

    return a_out, σ_out
```

Then we use ``jax.jit`` to compile the vectorized $K$.

```{code-cell} ipython3
K_egm_jax = jax.jit(K_egm_vec, static_argnums=(3,))
```

Next we define the successive approximator ``egm_jax`` with ``jax.jit`` vectorzied $K$.

```{code-cell} ipython3
def egm_jax(model,        
            tol=1e-5,
            max_iter=1000,
            verbose=True,
            print_skip=25):

    # Unpack
    constants, sizes, arrays = model
    
    β, R = constants
    n, grid_size, y_size = sizes
    grid, y_grid, P = arrays
    
    # Set up loop
    σ_init = jnp.tile(grid, [n, 1]).T
    a_init = jnp.copy(σ_init)
    a_vec, σ_vec = a_init, σ_init
    
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        a_new, σ_new = K_egm_jax(a_vec, σ_vec, constants, sizes, arrays)    
        error = jnp.max(jnp.abs(σ_vec - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        a_vec, σ_vec = jnp.copy(a_new), jnp.copy(σ_new)

    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return a_new, σ_new
```

### Numba version EGM

Here is the code for solving the same model with numba version EGM.

We will use the code for a sanity check on the results from the jax version EGM.

```{code-cell} ipython3
ifp_data = [
    ('R', float64),              # Interest rate 1 + r
    ('β', float64),              # Discount factor
    ('γ', float64),              # Preference parameter
    ('P', float64[:, :]),        # Markov matrix for binary Z_t
    ('y_grid', float64[:]),       # Income is Y_t = y[Z_t]
    ('grid', float64[:])         # Grid (array)
]

@jitclass(ifp_data)
class IFP:

    def __init__(self,
                 r=0.01,
                 β=0.96,
                 γ=1.5,
                 P=np.array(P),
                 y_grid=np.array(y_grid),
                 grid_max=16,
                 grid_size=200):

        self.R = 1 + r
        self.β, self.γ = β, γ

        
        self.P, self.y_grid = P, y_grid

        self.grid = np.linspace(0, grid_max, grid_size)

        # Recall that we need R β < 1 for convergence.
        assert self.R * self.β < 1, "Stability condition violated."

    def u_prime(self, c):
        return c**(-self.γ)
    
    def u_prime_inv(self, u_prime):
        return u_prime**(-1/self.γ)
```

```{code-cell} ipython3
@njit
def K_egm_nb(a_in, σ_in, ifp):
    """
    The operator K using EGM and numba.

    """
    
    # Simplify names
    R, P, y_grid, β, γ  = ifp.R, ifp.P, ifp.y_grid, ifp.β, ifp.γ
    grid, u_prime = ifp.grid, ifp.u_prime
    u_prime_inv = ifp.u_prime_inv
    n = len(P)
    
    
    # Linear interpolation of policy using endogenous grid
    def σ(a, z):
        return interp(a_in[:, z], σ_in[:, z], a)
    
    
    # Allocate memory for new consumption array
    σ_out = np.empty_like(σ_in)
    
    for i, s in enumerate(grid):
        for z in range(n):
            expect = 0.0
            for z_hat in range(n):
                expect += u_prime(σ(R * s + y_grid[z_hat], z_hat)) * P[z, z_hat]

            σ_out[i, z] = u_prime_inv(β * R * expect)
    
    # Calculate endogenous asset grid
    a_out = np.empty_like(σ_out)
    for z in range(n):
        a_out[:, z] = grid + σ_out[:, z]

    # Fixing a consumption-asset pair at (0, 0) improves interpolation
    σ_out[0, :] = 0
    a_out[0, :] = 0
    
    return a_out, σ_out
```

```{code-cell} ipython3
def egm_numba(model,        # Class with model information
              tol=1e-5,
              max_iter=1000,
              verbose=True,
              print_skip=25):

    # Unpack
    P, grid = model.P, model.grid
    n = len(P)
    
    σ_init = np.tile(grid, (n, 1)).T
    a_init = np.copy(σ_init)
    a_vec, σ_vec = a_init, σ_init
    
    # Set up loop
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        a_new, σ_new = K_egm_nb(a_vec, σ_vec, model)
        error = np.max(np.abs(σ_vec - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        a_vec, σ_vec = np.copy(a_new), np.copy(σ_new)

    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return a_new, σ_new
```

## A sanity check

First solve IFP with jax version EGM.

```{code-cell} ipython3
m_jax = ifp()
```

```{code-cell} ipython3
%%time
a_star_egm_jax, σ_star_egm_jax = egm_jax(m_jax,
                                         print_skip=5)
```

Then solve the same IFP with numba version EGM.

```{code-cell} ipython3
m_numba = IFP()
```

```{code-cell} ipython3
%%time
a_star_egm_nb, σ_star_egm_nb = egm_numba(m_numba,
                                         print_skip=5)
```

Finally we plot them for a sanity check.

```{code-cell} ipython3
fig, ax = plt.subplots()

n = len(m_numba.P)
for z in range(0, n-22):
    ax.plot(a_star_egm_nb[:, z], σ_star_egm_nb[:, z], label=f"numba EGM: consumption when $z={z}$")
    ax.plot(a_star_egm_jax[:, z], σ_star_egm_jax[:, z], label=f"jax EGM: consumption when $z={z}$")

ax.set_xlabel('asset')
plt.legend()
plt.show()
```

```{code-cell} ipython3

```
