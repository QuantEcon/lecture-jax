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

# Segregation with Persistent Shocks

## Overview

In previous lectures, we saw that the Schelling model converges to a
segregated equilibrium: agents relocate until everyone is happy, and then
the system stops.

But real cities don't work this way. People move in and out, neighborhoods
change, and the population is constantly being reshuffled by small shocks.

In this lecture, we explore what happens when we add this kind of persistent
randomness to the model.

Specifically, after each iteration, we randomly flip
the type of some agents with a small probability. 

We can interpret this as
agents occasionally moving away and being replaced by new agents whose type is
randomly determined.

With persistent shocks, the system never
converges, so the segregation dynamics keep operating indefinitely. 

Because agents are constantly being nudged out of equilibrium, the forces that drive
segregation never shut off. 

The result is that segregation levels **continue to
increase over time**, reaching levels **beyond what we see in the basic model**.

We use the parallel JAX implementation for efficiency, allowing us to run
longer simulations with more agents.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial
from typing import NamedTuple
import time
```

## Parameters

We use 2000 agents of each type and add a `flip_prob` parameter that controls
the probability of an agent's type being flipped after each iteration.

```{code-cell} ipython3
class Params(NamedTuple):
    num_of_type_0: int = 2000    # number of agents of type 0 (orange)
    num_of_type_1: int = 2000    # number of agents of type 1 (green)
    num_neighbors: int = 10      # number of neighbors
    max_other_type: int = 6      # max number of different-type neighbors tolerated
    num_candidates: int = 3      # candidate locations per agent per iteration
    flip_prob: float = 0.01      # probability of flipping an agent's type


params = Params()
```

## Setup

The following functions are repeated from the
{doc}`previous lecture <schelling_jax_parallel>`:

```{code-cell} ipython3
:tags: [hide-input]

def initialize_state(key, params):
    n = params.num_of_type_0 + params.num_of_type_1
    locations = random.uniform(key, (n, 2))
    types = jnp.concatenate([jnp.zeros(params.num_of_type_0, dtype=int),
                              jnp.ones(params.num_of_type_1, dtype=int)])
    return locations, types


@partial(jit, static_argnames=('params',))
def is_happy(loc, agent_idx, locations, types, params):
    " True if an agent at loc has at most max_other_type different-type neighbors. "
    distances = jnp.sum((loc - locations)**2, axis=1)
    distances = distances.at[agent_idx].set(jnp.inf)
    _, neighbors = jax.lax.top_k(-distances, params.num_neighbors)
    num_other = jnp.sum(types[neighbors] != types[agent_idx])
    return num_other <= params.max_other_type


@partial(jit, static_argnames=('params',))
def get_unhappy_agents(locations, types, params):
    " Return a boolean array indicating which agents are unhappy. "
    n = params.num_of_type_0 + params.num_of_type_1

    def is_unhappy(i):
        return ~is_happy(locations[i], i, locations, types, params)

    return vmap(is_unhappy)(jnp.arange(n))


@partial(jit, static_argnames=('params',))
def update_agent_location(i, locations, types, key, params):
    """
    Consider current location and num_candidates random alternatives.
    Return the first happy one. Already happy agents stay put.
    """
    current_loc = locations[i, :]

    # Build candidate list: current location + num_candidates random ones
    random_locs = random.uniform(key, (params.num_candidates, 2))
    candidates = jnp.vstack([current_loc[None, :], random_locs])

    # Check happiness at each candidate (in parallel)
    def check_candidate(loc):
        return is_happy(loc, i, locations, types, params)
    happy_at = vmap(check_candidate)(candidates)

    # Take the first happy candidate, or stay put if none are happy
    first_happy_idx = jnp.argmax(happy_at)
    return jnp.where(jnp.any(happy_at),
                     candidates[first_happy_idx],
                     current_loc)


@partial(jit, static_argnames=('params',))
def parallel_update_step(locations, types, key, params):
    """
    One step of the parallel algorithm: for each agent, find a happy
    candidate location (in parallel). Happy agents stay put, unhappy
    agents search for new locations.
    """
    n = params.num_of_type_0 + params.num_of_type_1

    keys = random.split(key, n + 1)
    key = keys[0]
    agent_keys = keys[1:]

    def update_one_agent(i):
        return update_agent_location(i, locations, types, agent_keys[i], params)
    new_locations = vmap(update_one_agent)(jnp.arange(n))

    return new_locations, key
```

## Type Flipping

This is the key addition in this lecture. After each iteration, we randomly
flip the type of each agent with probability `flip_prob`.

```{code-cell} ipython3
@partial(jit, static_argnames=('params',))
def flip_types(types, key, params):
    """
    Randomly flip agent types with probability flip_prob.
    """
    n = params.num_of_type_0 + params.num_of_type_1
    flip_prob = params.flip_prob

    # Generate random numbers for each agent
    random_vals = random.uniform(key, n)

    # Determine which agents get flipped
    should_flip = random_vals < flip_prob

    # Flip: 0 -> 1, 1 -> 0  (equivalent to 1 - type)
    flipped_types = 1 - types

    # Apply flips only where should_flip is True
    new_types = jnp.where(should_flip, flipped_types, types)

    return new_types
```

```{code-cell} ipython3
:tags: [hide-input]

def plot_distribution(locations, types, title):
    " Plot the distribution of agents. "
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

## Simulation with Shocks

The simulation loop now includes type flipping after each iteration. We run
for a fixed number of iterations rather than waiting for convergence, since
the system will never fully converge with ongoing shocks.

```{code-cell} ipython3
def run_simulation_with_shocks(params, max_iter=1000, seed=42, plot_every=100):
    """
    Run the Schelling simulation with random type flips.

    Parameters
    ----------
    params : Params
        Model parameters including flip_prob.
    max_iter : int
        Number of iterations to run.
    seed : int
        Random seed.
    plot_every : int
        Plot the distribution every this many iterations.
    """
    key = random.key(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key, params)
    n = locations.shape[0]

    print(f"Running simulation with {n} agents for {max_iter} iterations")
    print(f"Flip probability: {params.flip_prob}")
    print()

    plot_distribution(locations, types, 'Initial distribution')

    start_time = time.time()

    for iteration in range(1, max_iter + 1):
        # Update locations (agents try to find happy spots)
        locations, key = parallel_update_step(locations, types, key, params)

        # Apply random type flips
        key, flip_key = random.split(key)
        types = flip_types(types, flip_key, params)

        # Periodically report progress and plot
        if iteration % plot_every == 0:
            unhappy = get_unhappy_agents(locations, types, params)
            elapsed = time.time() - start_time
            print(f'Iteration {iteration}: {int(jnp.sum(unhappy))} unhappy agents, {elapsed:.1f}s elapsed')
            plot_distribution(locations, types, f'Iteration {iteration}')

    elapsed = time.time() - start_time
    print(f'\nCompleted {max_iter} iterations in {elapsed:.2f} seconds.')

    return locations, types
```

## Results

Let's warm up the JIT-compiled functions and run the simulation:

```{code-cell} ipython3
key = random.key(0)
key, init_key = random.split(key)
test_locations, test_types = initialize_state(init_key, params)

_ = is_happy(test_locations[0], 0, test_locations, test_types, params)
_ = get_unhappy_agents(test_locations, test_types, params)
key, subkey = random.split(key)
_ = update_agent_location(0, test_locations, test_types, subkey, params)
key, subkey = random.split(key)
_, _ = parallel_update_step(test_locations, test_types, subkey, params)
key, subkey = random.split(key)
_ = flip_types(test_types, subkey, params)

print("JAX functions compiled and ready!")
```

```{code-cell} ipython3
locations, types = run_simulation_with_shocks(params, max_iter=1000, plot_every=200)
```

## Discussion

The figures show an interesting result: segregation levels at the end of the
simulation are much higher than in the basic model without shocks.

Why does this happen?

In the basic model, the system converges to an equilibrium where everyone is
happy, and then the dynamics stop.

With persistent shocks, the system never converges — random type flips create
local pockets of unhappiness, triggering relocations that can cascade through
the population.

The key insight is that the segregation dynamics never shut off.

The result is that segregation continues to increase over time, reaching levels
far beyond what we observe when the system is allowed to converge.

This is arguably more realistic than the static equilibrium of the basic model.

Real cities experience constant population turnover, and the Schelling dynamics
operate continuously on the evolving population.
