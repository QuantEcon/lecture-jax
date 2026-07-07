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

# Schelling Model with JAX

## Overview

In the {doc}`previous lecture <schelling_numpy>`, we rewrote our Schelling model
using NumPy arrays and functions.

In this lecture, we explore [JAX](https://github.com/google/jax), a library
developed by Google for high-performance numerical computing.

JAX offers several powerful features, including automatic GPU/TPU acceleration and 
just-in-time compilation.

JAX is heavily used for AI workflows but we repurpose it to work with our simulation.

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from typing import NamedTuple
import time
```

## Setup

We use the same parameters as before. To keep our functions pure, we pack all
parameters into a `NamedTuple` that gets passed to functions that need them:

```{code-cell} ipython3
class Params(NamedTuple):
    num_of_type_0: int = 1000    # number of agents of type 0 (orange)
    num_of_type_1: int = 1000    # number of agents of type 1 (green)
    num_neighbors: int = 10      # number of agents regarded as neighbors
    max_other_type: int = 6     # max number of different-type neighbors tolerated


params = Params()
```

Here's our initialization function. Note that we use `jax.random` instead of
`numpy.random` — we pass a `key` argument to `random.uniform`, making random
generation deterministic and reproducible:

```{code-cell} ipython3
def initialize_state(key, params):
    """
    Initialize agent locations and types.

    """
    num_of_type_0, num_of_type_1 = params.num_of_type_0, params.num_of_type_1
    n = num_of_type_0 + num_of_type_1
    locations = random.uniform(key, (n, 2))
    types = jnp.concatenate([jnp.zeros(num_of_type_0, dtype=int),
                              jnp.ones(num_of_type_1, dtype=int)])
    return locations, types
```

## JAX-Compiled Functions

Now let's rewrite our core functions for JAX.

We use `jit` to compile functions for faster execution.

```{code-cell} ipython3
@partial(jit, static_argnames=('params',))
def is_happy(loc, agent_idx, locations, types, params):
    " True if an agent at loc has at most max_other_type different-type neighbors. "
    # Squared distances from loc to every agent
    distances = jnp.sum((loc - locations)**2, axis=1)
    distances = distances.at[agent_idx].set(jnp.inf)  # exclude self
    # top_k finds the k smallest distances in O(n) (we negate to use top_k)
    _, neighbors = jax.lax.top_k(-distances, params.num_neighbors)
    num_other = jnp.sum(types[neighbors] != types[agent_idx])
    return num_other <= params.max_other_type
```

Compared to the NumPy version, there are a few differences worth noting.

We use `jax.lax.top_k` instead of `argsort` to find nearest neighbors — this
is $O(n)$ rather than $O(n \log n)$.

We use `.at[].set()` rather than direct indexing to exclude the agent from its
own neighbor set, since JAX arrays are immutable.

The function takes `loc` as an explicit argument rather than looking it up
from the arrays, so we can test hypothetical locations without modifying the
`locations` array.


The next function finds a location where a given agent would be happy.

Rather than updating the `locations` array on each iteration, it tests
candidate locations directly and returns only the final location.

```{code-cell} ipython3
@partial(jit, static_argnames=('params',))
def move_agent(i, locations, types, key, params, max_attempts=10_000):
    """
    Find a location where agent i is happy.

    Returns the new location and updated random key. The calling code
    is responsible for updating the locations array if the agent moved.
    """
    loc = locations[i, :]

    # Continue while under max_attempts and not yet happy
    def while_test(state):
        loc, key, attempts = state
        return (attempts < max_attempts) & ~is_happy(loc, i, locations, types, params)

    # Draw a new random location
    def update(state):
        _, key, attempts = state
        key, subkey = random.split(key)
        new_loc = random.uniform(subkey, 2)
        return new_loc, key, attempts + 1

    final_loc, key, _ = jax.lax.while_loop(while_test, update, (loc, key, 0))
    return final_loc, key
```

Here `jax.lax.while_loop` calls `update` repeatedly until `while_test` returns False.

## The Simulation

```{code-cell} ipython3
:tags: [hide-input]

def plot_distribution(locations, types, title):
    """
    Plot the distribution of agents.
    """
    # Convert JAX arrays to NumPy for matplotlib
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

```{code-cell} ipython3
def simulation_loop(locations, types, key, params, max_iter):
    """
    Run the simulation loop until convergence or max iterations.
    """
    n = params.num_of_type_0 + params.num_of_type_1
    converged = False
    for iteration in range(1, max_iter + 1):
        print(f'Entering iteration {iteration}')
        someone_moved = False
        for i in range(n):
            if not is_happy(locations[i], i, locations, types, params):
                new_loc, key = move_agent(i, locations, types, key, params)
                locations = locations.at[i, :].set(new_loc)
                someone_moved = True
        if not someone_moved:
            converged = True
            break

    return locations, iteration, converged, key
```

```{code-cell} ipython3
def run_simulation(params, max_iter=100_000, seed=1234):
    """
    Run the Schelling simulation using JAX.
    """
    key = random.key(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key, params)

    plot_distribution(locations, types, 'Initial distribution')

    start_time = time.time()
    locations, iteration, converged, key = simulation_loop(locations, types, key, params, max_iter)
    elapsed = time.time() - start_time

    plot_distribution(locations, types, f'Iteration {iteration}')

    if converged:
        print(f'Converged in {elapsed:.2f} seconds after {iteration} iterations.')
    else:
        print('Hit iteration bound and terminated.')

    return locations, types
```

The simulation loop is similar to the NumPy version: it cycles through agents,
checks each one for happiness, and moves the unhappy ones.

(schelling_jax_results)=
## Results

JAX compiles functions the first time they're called. Let's warm them up
before timing the simulation:

```{code-cell} ipython3
key = random.key(42)
key, init_key = random.split(key)
test_locations, test_types = initialize_state(init_key, params)

_ = is_happy(test_locations[0], 0, test_locations, test_types, params)
key, subkey = random.split(key)
_, _ = move_agent(0, test_locations, test_types, subkey, params)

print("JAX functions compiled and ready!")
```

Now let's run the simulation:

```{code-cell} ipython3
locations, types = run_simulation(params)
```

## Limitations of This Approach

While this lecture demonstrated JAX syntax and concepts, the algorithm itself
doesn't fully leverage JAX's parallel capabilities. The original Schelling
algorithm has inherent sequential dependencies:

- Agents update one at a time
- Each agent's move changes the state for subsequent agents
- The "move until happy" while loop has unpredictable length

These characteristics don't map well to parallel hardware like GPUs, which
excel at performing the same operation on many data points simultaneously.


In the {doc}`next lecture <schelling_jax_parallel>`, we'll restructure the
algorithm to better leverage JAX's parallel capabilities.
