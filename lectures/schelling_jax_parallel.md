---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Parallelizing the Algorithm

## Overview

In the previous lectures, we implemented the Schelling segregation model using:

1. {doc}`NumPy arrays and functions <schelling_numpy>`
2. {doc}`JAX with JIT compilation <schelling_jax>`

NumPy offered speed gains from vectorization.

JAX was slightly faster, with some small amount of parallelization achieved.

Parallelization was limited however, because the algorithm is heavily
sequential.

In this lecture,  introduce a **parallel algorithm** that 

* is in some sense less elegant but
* fully leverages JAX's ability to perform vectorized operations across all agents simultaneously.

Even though the algorithm is less elegant, it still converges in a relatively
small number of steps.

Moreover, the parallel nature of the algorithm allows us to exploit the full
power of JAX.

Our plan for the lecture is to compare three implementations

1. The original NumPy one,
1. The original JAX one, and
1. The new parallelized JAX algorithm.

We'll run a "horse race" to see how each approach performs.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from typing import NamedTuple
from numpy.random import uniform
import time
```

## Parameters

We use the same parameters across all implementations. To keep our functions
pure, we pack all parameters into a `NamedTuple` that gets passed to functions
that need them:

```{code-cell} ipython3
class Params(NamedTuple):
    num_of_type_0: int = 1800    # number of agents of type 0 (orange)
    num_of_type_1: int = 1800    # number of agents of type 1 (green)
    num_neighbors: int = 10      # number of neighbors
    max_other_type: int = 6     # max number of different-type neighbors tolerated
    num_candidates: int = 3      # candidate locations per agent per iteration


params = Params()
```

## Shared Plotting Function

```{code-cell} ipython3
def plot_distribution(locations, types, title):
    " Plot the distribution of agents. "
    # Convert to NumPy if needed (for JAX arrays)
    locations_np = np.asarray(locations)
    types_np = np.asarray(types)

    fig, ax = plt.subplots()
    plot_args = {'markersize': 6, 'alpha': 0.8, 'markeredgecolor': 'black', 'markeredgewidth': 0.5}
    colors = 'darkorange', 'green'
    for agent_type, color in zip((0, 1), colors):
        idx = (types_np == agent_type)
        ax.plot(locations_np[idx, 0],
                locations_np[idx, 1],
                'o',
                markerfacecolor=color,
                **plot_args)
    ax.set_title(title)
    plt.show()
```

## NumPy Implementation

First, let's copy the NumPy version from {doc}`schelling_numpy`:

```{code-cell} ipython3
def np_initialize_state(params):
    num_of_type_0, num_of_type_1 = params.num_of_type_0, params.num_of_type_1
    n = num_of_type_0 + num_of_type_1
    locations = uniform(size=(n, 2))
    types = np.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types


def np_get_distances(loc, locations):
    return np.linalg.norm(loc - locations, axis=1)


def np_get_neighbors(i, locations, params):
    num_neighbors = params.num_neighbors
    loc = locations[i, :]
    distances = np_get_distances(loc, locations)
    distances[i] = np.inf
    indices = np.argsort(distances)
    return indices[:num_neighbors]


def np_is_happy(i, locations, types, params):
    max_other_type = params.max_other_type
    agent_type = types[i]
    neighbors = np_get_neighbors(i, locations, params)
    neighbor_types = types[neighbors]
    num_other = np.sum(neighbor_types != agent_type)
    return num_other <= max_other_type


def np_update_agent(i, locations, types, params, max_attempts=10_000):
    attempts = 0
    while not np_is_happy(i, locations, types, params):
        locations[i, :] = uniform(), uniform()
        attempts += 1
        if attempts >= max_attempts:
            break


def run_numpy_simulation(params, max_iter=100_000, seed=42):
    n = params.num_of_type_0 + params.num_of_type_1
    np.random.seed(seed)
    locations, types = np_initialize_state(params)

    plot_distribution(locations, types, 'NumPy: Initial distribution')

    start_time = time.time()
    someone_moved = True
    iteration = 0
    while someone_moved and iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1
        someone_moved = False
        for i in range(n):
            if not np_is_happy(i, locations, types, params):
                np_update_agent(i, locations, types, params)
                someone_moved = True
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'NumPy: Iteration {iteration}')

    if not someone_moved:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## JAX Sequential Implementation

Next, we copy the JAX version from {doc}`schelling_jax`:

```{code-cell} ipython3
def jax_initialize_state(key, params):
    num_of_type_0, num_of_type_1 = params.num_of_type_0, params.num_of_type_1
    n = num_of_type_0 + num_of_type_1
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * num_of_type_0 + [1] * num_of_type_1)
    return locations, types


@jit
def jax_get_distances(loc, locations):
    diff = loc - locations
    return jnp.sum(diff**2, axis=1)


@partial(jit, static_argnames=('params',))
def jax_get_neighbors(loc, agent_idx, locations, params):
    num_neighbors = params.num_neighbors
    distances = jax_get_distances(loc, locations)
    distances = distances.at[agent_idx].set(jnp.inf)
    _, indices = jax.lax.top_k(-distances, num_neighbors)
    return indices


@partial(jit, static_argnames=('params',))
def jax_is_unhappy(loc, agent_type, agent_idx, locations, types, params):
    max_other_type = params.max_other_type
    neighbors = jax_get_neighbors(loc, agent_idx, locations, params)
    neighbor_types = types[neighbors]
    num_other = jnp.sum(neighbor_types != agent_type)
    return num_other > max_other_type


@partial(jit, static_argnames=('params',))
def jax_update_agent(i, locations, types, key, params, max_attempts=10_000):
    loc = locations[i, :]
    agent_type = types[i]

    def cond_fn(state):
        loc, key, attempts = state
        return (attempts < max_attempts) & jax_is_unhappy(loc, agent_type, i, locations, types, params)

    def body_fn(state):
        _, key, attempts = state
        key, subkey = random.split(key)
        new_loc = random.uniform(subkey, shape=(2,))
        return new_loc, key, attempts + 1

    final_loc, key, _ = jax.lax.while_loop(cond_fn, body_fn, (loc, key, 0))
    return final_loc, key


@partial(jit, static_argnames=('params',))
def jax_get_unhappy_agents(locations, types, params):
    n = params.num_of_type_0 + params.num_of_type_1

    def check_agent(i):
        return jax_is_unhappy(locations[i], types[i], i, locations, types, params)

    all_unhappy = vmap(check_agent)(jnp.arange(n))
    # jnp.where with size= returns fixed-length array (required for JIT)
    # Pads with fill_value=-1 when fewer than n agents are unhappy
    indices = jnp.where(all_unhappy, size=n, fill_value=-1)[0]
    count = jnp.sum(all_unhappy)  # number of valid indices
    return indices, count


def jax_simulation_loop(locations, types, key, params, max_iter):
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        unhappy, num_unhappy = jax_get_unhappy_agents(locations, types, params)

        if num_unhappy == 0:
            break

        for j in range(int(num_unhappy)):
            i = int(unhappy[j])
            new_loc, key = jax_update_agent(i, locations, types, key, params)
            locations = locations.at[i, :].set(new_loc)

    return locations, iteration, key


def run_jax_simulation(params, max_iter=100_000, seed=42):
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = jax_initialize_state(init_key, params)

    plot_distribution(locations, types, 'JAX Sequential: Initial distribution')

    start_time = time.time()
    locations, iteration, key = jax_simulation_loop(locations, types, key, params, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'JAX Sequential: Iteration {iteration}')

    if iteration < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## JAX Parallel Implementation

Now we introduce the parallel algorithm. 

Our aim is to update all agents at the same time, rather than sequentially.

To do this we

1. **Identify all unhappy agents** in parallel
2. **Generate candidate locations** for all unhappy agents in parallel
3. **Test happiness** at all candidate locations in parallel
4. **Update all agents** simultaneously

Moreover, when we generate candidate locations, we will offer a fixed number to
all agents.

This allows the parallel threads to do the same amount of work, so they all run
at the same speed.

This approach is well-suited to GPU execution, where thousands of operations
can run concurrently.

### Trade-off I

The sequential algorithm guarantees that each agent finds a happy location
before moving on. 

The parallel algorithm instead proposes a fixed number of candidate locations
per agent per iteration. 

If none of the candidates make the agent happy, the agent stays put and tries again next iteration.

This means the parallel algorithm may need more iterations. 

However, each iteration is faster because all work is done in parallel.

### Trade-off II

Because we update all agents at once, the agents have less information --- they
are predicting the next period distribution from the current one.

(All agents take the current distribution of agents as their information, rather
than waiting until other agents update and viewing the true distribution.)

We hope that, nonetheless, the algorithm will converge.

### Core Parallel Functions

The `update_agent_location` function below performs all computation (generating
candidates, checking happiness at each candidate) upfront before making the
final decision about whether to move. 

This may seem wasteful for agents who are
already happy, but it's actually optimal for parallel execution.

In SIMD/SIMT architectures (GPUs, vectorized CPU operations), all threads
execute the same instructions in lockstep. Conditional branches like
`jax.lax.cond` don't skip work—both branches are computed and the result is
selected afterward. 

By doing uniform work for all agents and using `jnp.where`
to select results at the end, we align with how the hardware actually executes
the code.

```{code-cell} ipython3
@partial(jit, static_argnames=('params',))
def update_agent_location(i, locations, types, key, params):
    """
    Propose num_candidates random locations for agent i.
    Return the first happy candidate if agent is unhappy, otherwise current location.
    """
    num_candidates = params.num_candidates
    current_loc = locations[i, :]
    agent_type = types[i]

    # Generate num_candidates random locations
    keys = random.split(key, num_candidates)
    candidates = vmap(lambda k: random.uniform(k, shape=(2,)))(keys)

    # Check happiness at each candidate location (in parallel)
    def check_candidate(loc):
        return ~jax_is_unhappy(loc, agent_type, i, locations, types, params)
    happy_at_candidates = vmap(check_candidate)(candidates)

    # Find first happy candidate (if any)
    first_happy_idx = jnp.argmax(happy_at_candidates)
    any_happy = jnp.any(happy_at_candidates)

    # Check if agent is already happy at current location
    is_happy = ~jax_is_unhappy(current_loc, agent_type, i, locations, types, params)

    # Move only if unhappy and found a happy candidate; otherwise stay put
    new_loc = jnp.where(is_happy,
                current_loc,                      # Happy agents branch
                jnp.where(                        # Unhappy agents branch
                    any_happy,                    # If there is a good candidate 
                    candidates[first_happy_idx],  # Move to it
                    current_loc                   # Otherwise stay still
                )
              )
    return new_loc


@partial(jit, static_argnames=('params',))
def parallel_update_step(locations, types, key, params):
    """
    One step of the parallel algorithm:
    1. Generate keys for all agents
    2. For each agent, find a happy candidate location (in parallel)
       (happy agents stay put, unhappy agents search for new locations)
    """
    n = params.num_of_type_0 + params.num_of_type_1

    # Generate keys for all agents
    keys = random.split(key, n + 1)
    key = keys[0]
    agent_keys = keys[1:]

    # For each agent, find a happy candidate location (in parallel)
    def update_one_agent(i):
        return update_agent_location(i, locations, types, agent_keys[i], params)
    new_locations = vmap(update_one_agent)(jnp.arange(n))

    return new_locations, key
```

### Parallel Simulation Loop

```{code-cell} ipython3
def parallel_simulation_loop(locations, types, key, params, max_iter):
    iteration = 0
    while iteration < max_iter:
        print(f'Entering iteration {iteration + 1}')
        iteration += 1

        _, num_unhappy = jax_get_unhappy_agents(locations, types, params)

        if num_unhappy == 0:
            break

        locations, key = parallel_update_step(locations, types, key, params)

    return locations, iteration, key


def run_parallel_simulation(params, max_iter=100_000, seed=42):
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = jax_initialize_state(init_key, params)

    plot_distribution(locations, types, 'JAX Parallel: Initial distribution')

    start_time = time.time()
    locations, iteration, key = parallel_simulation_loop(locations, types, key, params, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'JAX Parallel: Iteration {iteration}')

    if iteration < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

## Warming Up JAX

Before timing, we compile all JAX functions:

```{code-cell} ipython3
key = random.PRNGKey(0)
key, init_key = random.split(key)
test_locations, test_types = jax_initialize_state(init_key, params)

# Warm up JAX sequential functions
_ = jax_get_distances(test_locations[0], test_locations)
_ = jax_get_neighbors(test_locations[0], 0, test_locations, params)
_ = jax_is_unhappy(test_locations[0], test_types[0], 0, test_locations, test_types, params)
_, _ = jax_get_unhappy_agents(test_locations, test_types, params)
key, subkey = random.split(key)
_, _ = jax_update_agent(0, test_locations, test_types, subkey, params)

# Warm up JAX parallel functions
key, subkey = random.split(key)
_ = update_agent_location(0, test_locations, test_types, subkey, params)
key, subkey = random.split(key)
_, _ = parallel_update_step(test_locations, test_types, subkey, params)

print("JAX functions compiled and ready!")
```

## The Horse Race

Now let's run all three implementations and compare their performance.

### NumPy

```{code-cell} ipython3
print("=" * 50)
print("NUMPY")
print("=" * 50)
locations_np, types_np = run_numpy_simulation(params)
```

### JAX Sequential

```{code-cell} ipython3
print("=" * 50)
print("JAX SEQUENTIAL")
print("=" * 50)
locations_jax, types_jax = run_jax_simulation(params)
```

### JAX Parallel

```{code-cell} ipython3
print("=" * 50)
print("JAX PARALLEL")
print("=" * 50)
locations_par, types_par = run_parallel_simulation(params)
```

## Discussion

The results reveal interesting trade-offs:

1. **NumPy** provides a straightforward implementation but runs entirely on
   the CPU with Python loops.

2. **JAX Sequential** uses JIT compilation for individual operations, but the
   outer loop still processes agents one at a time.

3. **JAX Parallel** processes all agents simultaneously each iteration. While
   it may require more iterations (since agents might not find a happy location
   in their limited candidates), each iteration leverages massive parallelism.

The parallel approach shines on GPUs, where thousands of threads can evaluate
candidate locations concurrently. On CPUs, the benefits are more modest, but
the parallel structure still allows JAX to optimize memory access patterns and
use SIMD instructions effectively.

## Key Takeaways

1. **Algorithm structure matters**: Simply porting code to JAX doesn't
   automatically make it faster. To fully benefit from JAX's capabilities,
   algorithms often need to be restructured for parallelism.

2. **Trade iteration count for parallelism**: The parallel algorithm may need
   more iterations, but each iteration does more work in parallel. This
   trade-off often favors parallelism on modern hardware.

3. **GPU acceleration**: The parallel algorithm is particularly well-suited
   for GPUs, where the speedup can be dramatic. On CPU-only systems, the
   difference is smaller.
