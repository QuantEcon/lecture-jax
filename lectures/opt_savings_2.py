# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Optimal Savings II: Alternative Algorithms
#
# ```{include} _admonition/gpu.md
# ```
#
# In {doc}`opt_savings_1` we solved a simple version of the household optimal
# savings problem via value function iteration (VFI) using JAX.
#
# In this lecture we tackle exactly the same problem while adding in two
# alternative algorithms:
#
# * optimistic policy iteration (OPI) and
# * Howard policy iteration (HPI).
#
# We will see that both of these algorithms outperform traditional VFI.
#
# One reason for this is that the algorithms have good convergence properties.
#
# Another is that one of them, HPI, is particularly well suited to pairing with
# JAX.
#
# The reason is that HPI uses a relatively small number of computationally expensive steps,
# whereas VFI uses a longer sequence of small steps.
#
# In other words, VFI is inherently more sequential than HPI, and sequential
# routines are hard to parallelize.
#
# By comparison, HPI is less sequential -- the small number of computationally
# intensive steps can be effectively parallelized by JAX.
#
# This is particularly valuable when the underlying hardware includes a GPU.
#
# Details on VFI, HPI and OPI can be found in [this book](https://dp.quantecon.org), for which a PDF is freely available.
#
# Here we assume readers have some knowledge of the algorithms and focus on
# computation.
#
# For the details of the savings model, readers can refer to {doc}`opt_savings_1`.
#
# In addition to what’s in Anaconda, this lecture will need the following libraries:

# + tags=["hide-output"]
# !pip install quantecon
# -

# We will use the following imports:

import quantecon as qe
import jax
import jax.numpy as jnp
from collections import namedtuple
import matplotlib.pyplot as plt
import time

# Let's check the GPU we are running.

# !nvidia-smi

# We'll use 64 bit floats to gain extra precision.

jax.config.update("jax_enable_x64", True)


# ## Model primitives
#
# First we define a model that stores parameters and grids.
#
# The {ref}`following code <prgm:create-consumption-model>` is repeated from {doc}`opt_savings_1`.

def create_consumption_model(R=1.01,                    # Gross interest rate
                             β=0.98,                    # Discount factor
                             γ=2,                       # CRRA parameter
                             w_min=0.01,                # Min wealth
                             w_max=5.0,                 # Max wealth
                             w_size=150,                # Grid side
                             ρ=0.9, ν=0.1, y_size=100): # Income parameters
    """
    A function that takes in parameters and returns parameters and grids 
    for the optimal savings problem.
    """
    w_grid = jnp.linspace(w_min, w_max, w_size)
    mc = qe.tauchen(n=y_size, rho=ρ, sigma=ν)
    y_grid, Q = jnp.exp(mc.state_values), jax.device_put(mc.P)
    sizes = w_size, y_size
    return (β, R, γ), sizes, (w_grid, y_grid, Q)


# Here's the right hand side of the Bellman equation:

def B(v, params, sizes, arrays):
    """
    A vectorized version of the right-hand side of the Bellman equation
    (before maximization), which is a 3D array representing

        B(w, y, w′) = u(Rw + y - w′) + β Σ_y′ v(w′, y′) Q(y, y′)

    for all (w, y, w′).
    """

    # Unpack
    β, R, γ = params
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    # Compute current rewards r(w, y, wp) as array r[i, j, ip]
    w  = jnp.reshape(w_grid, (w_size, 1, 1))    # w[i]   ->  w[i, j, ip]
    y  = jnp.reshape(y_grid, (1, y_size, 1))    # z[j]   ->  z[i, j, ip]
    wp = jnp.reshape(w_grid, (1, 1, w_size))    # wp[ip] -> wp[i, j, ip]
    c = R * w + y - wp

    # Calculate continuation rewards at all combinations of (w, y, wp)
    v = jnp.reshape(v, (1, 1, w_size, y_size))  # v[ip, jp] -> v[i, j, ip, jp]
    Q = jnp.reshape(Q, (1, y_size, 1, y_size))  # Q[j, jp]  -> Q[i, j, ip, jp]
    EV = jnp.sum(v * Q, axis=3)                 # sum over last index jp

    # Compute the right-hand side of the Bellman equation
    return jnp.where(c > 0, c**(1-γ)/(1-γ) + β * EV, -jnp.inf)


# ## Operators
#
# We define a function to compute the current rewards $r_\sigma$ given policy $\sigma$,
# which is defined as the vector
#
# $$
# r_\sigma(w, y) := r(w, y, \sigma(w, y)) 
# $$

def compute_r_σ(σ, params, sizes, arrays):
    """
    Compute the array r_σ[i, j] = r[i, j, σ[i, j]], which gives current
    rewards given policy σ.
    """

    # Unpack model
    β, R, γ = params
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    # Compute r_σ[i, j]
    w = jnp.reshape(w_grid, (w_size, 1))
    y = jnp.reshape(y_grid, (1, y_size))
    wp = w_grid[σ]
    c = R * w + y - wp
    r_σ = c**(1-γ)/(1-γ)

    return r_σ


# Now we define the policy operator $T_\sigma$

def T_σ(v, σ, params, sizes, arrays):
    "The σ-policy operator."

    # Unpack model
    β, R, γ = params
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    r_σ = compute_r_σ(σ, params, sizes, arrays)

    # Compute the array v[σ[i, j], jp]
    yp_idx = jnp.arange(y_size)
    yp_idx = jnp.reshape(yp_idx, (1, 1, y_size))
    σ = jnp.reshape(σ, (w_size, y_size, 1))
    V = v[σ, yp_idx]

    # Convert Q[j, jp] to Q[i, j, jp]
    Q = jnp.reshape(Q, (1, y_size, y_size))

    # Calculate the expected sum Σ_jp v[σ[i, j], jp] * Q[i, j, jp]
    EV = jnp.sum(V * Q, axis=2)

    return r_σ + β * EV


# and the Bellman operator $T$

def T(v, params, sizes, arrays):
    "The Bellman operator."
    return jnp.max(B(v, params, sizes, arrays), axis=2)


# The next function computes a $v$-greedy policy given $v$

def get_greedy(v, params, sizes, arrays):
    "Computes a v-greedy policy, returned as a set of indices."
    return jnp.argmax(B(v, params, sizes, arrays), axis=2)


# The function below computes the value $v_\sigma$ of following policy $\sigma$.
#
# This lifetime value is a function $v_\sigma$ that satisfies
#
# $$
# v_\sigma(w, y) = r_\sigma(w, y) + \beta \sum_{y'} v_\sigma(\sigma(w, y), y') Q(y, y')
# $$
#
# We wish to solve this equation for $v_\sigma$.
#
# Suppose we define the linear operator $L_\sigma$ by
#
# $$ 
# (L_\sigma v)(w, y) = v(w, y) - \beta \sum_{y'} v(\sigma(w, y), y') Q(y, y')
# $$
#
# With this notation, the problem is to solve for $v$ via
#
# $$
# (L_{\sigma} v)(w, y) = r_\sigma(w, y)
# $$
#
# In vector for this is $L_\sigma v = r_\sigma$, which tells us that the function
# we seek is
#
# $$ 
# v_\sigma = L_\sigma^{-1} r_\sigma 
# $$
#
# JAX allows us to solve linear systems defined in terms of operators; the first
# step is to define the function $L_{\sigma}$.

def L_σ(v, σ, params, sizes, arrays):
    """
    Here we set up the linear map v -> L_σ v, where 

        (L_σ v)(w, y) = v(w, y) - β Σ_y′ v(σ(w, y), y′) Q(y, y′)

    """

    β, R, γ = params
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    # Set up the array v[σ[i, j], jp]
    zp_idx = jnp.arange(y_size)
    zp_idx = jnp.reshape(zp_idx, (1, 1, y_size))
    σ = jnp.reshape(σ, (w_size, y_size, 1))
    V = v[σ, zp_idx]

    # Expand Q[j, jp] to Q[i, j, jp]
    Q = jnp.reshape(Q, (1, y_size, y_size))

    # Compute and return v[i, j] - β Σ_jp v[σ[i, j], jp] * Q[j, jp]
    return v - β * jnp.sum(V * Q, axis=2)


# Now we can define a function to compute $v_{\sigma}$

def get_value(σ, params, sizes, arrays):
    "Get the value v_σ of policy σ by inverting the linear map L_σ."

    # Unpack
    β, R, γ = params
    w_size, y_size = sizes
    w_grid, y_grid, Q = arrays

    r_σ = compute_r_σ(σ, params, sizes, arrays)

    # Reduce L_σ to a function in v
    partial_L_σ = lambda v: L_σ(v, σ, params, sizes, arrays)

    return jax.scipy.sparse.linalg.bicgstab(partial_L_σ, r_σ)[0]


# ## JIT compiled versions

B = jax.jit(B, static_argnums=(2,))
compute_r_σ = jax.jit(compute_r_σ, static_argnums=(2,))
T = jax.jit(T, static_argnums=(2,))
get_greedy = jax.jit(get_greedy, static_argnums=(2,))
get_value = jax.jit(get_value, static_argnums=(2,))
T_σ = jax.jit(T_σ, static_argnums=(3,))
L_σ = jax.jit(L_σ, static_argnums=(3,))


# We use successive approximation for VFI.

# +
def successive_approx_jax(T,                     # Operator (callable)
                          x_0,                   # Initial condition                
                          tol=1e-6,              # Error tolerance
                          max_iter=10_000):      # Max iteration bound
    def body_fun(k_x_err):
        k, x, error = k_x_err
        x_new = T(x)
        error = jnp.max(jnp.abs(x_new - x))
        return k + 1, x_new, error

    def cond_fun(k_x_err):
        k, x, error = k_x_err
        return jnp.logical_and(error > tol, k < max_iter)

    k, x, error = jax.lax.while_loop(cond_fun, body_fun, (1, x_0, tol + 1))
    return x

successive_approx_jax = jax.jit(successive_approx_jax, static_argnums=(0,))


# -

# For OPI we'll add a compiled routine that computes $T_σ^m v$.

# +
def iterate_policy_operator(σ, v, m, params, sizes, arrays):

    def update(i, v):
        v = T_σ(v, σ, params, sizes, arrays)
        return v
    
    v = jax.lax.fori_loop(0, m, update, v)
    return v

iterate_policy_operator = jax.jit(iterate_policy_operator,
                                  static_argnums=(4,))


# -

# ## Solvers
#
# Now we define the solvers, which implement VFI, HPI and OPI.
#
# Here's VFI.

def value_function_iteration(model, tol=1e-5):
    """
    Implements value function iteration.
    """
    params, sizes, arrays = model
    vz = jnp.zeros(sizes)
    _T = lambda v: T(v, params, sizes, arrays)
    v_star = successive_approx_jax(_T, vz, tol=tol)
    return get_greedy(v_star, params, sizes, arrays)


# Here's HPI.

def howard_policy_iteration(model, maxiter=250):
    """
    Implements Howard policy iteration (see dp.quantecon.org)
    """
    params, sizes, arrays = model
    σ = jnp.zeros(sizes, dtype=int)
    i, error = 0, 1.0
    while error > 0 and i < maxiter:
        v_σ = get_value(σ, params, sizes, arrays)
        σ_new = get_greedy(v_σ, params, sizes, arrays)
        error = jnp.max(jnp.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error {error}.")
    return σ


def optimistic_policy_iteration(model, tol=1e-5, m=10):
    """
    Implements optimistic policy iteration (see dp.quantecon.org)
    """
    params, sizes, arrays = model
    v = jnp.zeros(sizes)
    error = tol + 1
    while error > tol:
        last_v = v
        σ = get_greedy(v, params, sizes, arrays)
        v = iterate_policy_operator(σ, v, m, params, sizes, arrays)
        error = jnp.max(jnp.abs(v - last_v))
    return get_greedy(v, params, sizes, arrays)


# ## Plots
#
# Create a model for consumption, perform policy iteration, and plot the resulting optimal policy function.

model = create_consumption_model()
# Unpack
params, sizes, arrays = model
β, R, γ = params
w_size, y_size = sizes
w_grid, y_grid, Q = arrays

# + mystnb={"figure": {"caption": "Optimal policy function", "name": "optimal-policy-function"}}
σ_star = howard_policy_iteration(model)

fig, ax = plt.subplots()
ax.plot(w_grid, w_grid, "k--", label="45")
ax.plot(w_grid, w_grid[σ_star[:, 1]], label="$\\sigma^*(\cdot, y_1)$")
ax.plot(w_grid, w_grid[σ_star[:, -1]], label="$\\sigma^*(\cdot, y_N)$")
ax.legend()
plt.show()
# -

# ## Tests
#
# Here's a quick test of the timing of each solver.

model = create_consumption_model()

print("Starting HPI.")
start_time = time.time()
out = howard_policy_iteration(model)
elapsed = time.time() - start_time
print(f"HPI completed in {elapsed} seconds.")

print("Starting VFI.")
start_time = time.time()
out = value_function_iteration(model)
elapsed = time.time() - start_time
print(f"VFI completed in {elapsed} seconds.")

print("Starting OPI.")
start_time = time.time()
out = optimistic_policy_iteration(model, m=100)
elapsed = time.time() - start_time
print(f"OPI completed in {elapsed} seconds.")


def run_algorithm(algorithm, model, **kwargs):
    start_time = time.time()
    result = algorithm(model, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{algorithm.__name__} completed in {elapsed_time:.2f} seconds.")
    return result, elapsed_time


# +
model = create_consumption_model()
σ_pi, pi_time = run_algorithm(howard_policy_iteration, model)
σ_vfi, vfi_time = run_algorithm(value_function_iteration, model, tol=1e-5)

m_vals = range(5, 600, 40)
opi_times = []
for m in m_vals:
    σ_opi, opi_time = run_algorithm(optimistic_policy_iteration, 
                                    model, m=m, tol=1e-5)
    opi_times.append(opi_time)

# + mystnb={"figure": {"caption": "Solver times", "name": "howard+value+optimistic-solver-times"}}
fig, ax = plt.subplots()
ax.plot(m_vals, 
        jnp.full(len(m_vals), pi_time), 
        lw=2, label="Howard policy iteration")
ax.plot(m_vals, 
        jnp.full(len(m_vals), vfi_time), 
        lw=2, label="value function iteration")
ax.plot(m_vals, opi_times, 
        lw=2, label="optimistic policy iteration")
ax.legend(frameon=False)
ax.set_xlabel("$m$")
ax.set_ylabel("time")
plt.show()
