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
n = params.num_of_type_0 + params.num_of_type_1
```

## Setup Functions

We reuse the core functions from the parallel JAX implementation.

### Initialization

```{code-cell} ipython3
def initialize_state(key, params):
    n = params.num_of_type_0 + params.num_of_type_1
    locations = random.uniform(key, shape=(n, 2))
    types = jnp.array([0] * params.num_of_type_0 + [1] * params.num_of_type_1)
    return locations, types
```

### Distance and Neighbor Functions

```{code-cell} ipython3
@jit
def get_distances(loc, locations):
    diff = locations - loc
    return jnp.sum(diff**2, axis=1)


@partial(jit, static_argnames=('params',))
def get_neighbors(loc, agent_idx, locations, params):
    num_neighbors = params.num_neighbors
    distances = get_distances(loc, locations)
    distances = distances.at[agent_idx].set(jnp.inf)
    _, indices = jax.lax.top_k(-distances, num_neighbors)
    return indices


@partial(jit, static_argnames=('params',))
def is_unhappy(loc, agent_type, agent_idx, locations, types, params):
    max_other_type = params.max_other_type
    neighbors = get_neighbors(loc, agent_idx, locations, params)
    neighbor_types = types[neighbors]
    num_other = jnp.sum(neighbor_types != agent_type)
    return num_other > max_other_type


@partial(jit, static_argnames=('params',))
def get_unhappy_agents(locations, types, params):
    n = params.num_of_type_0 + params.num_of_type_1

    def check_agent(i):
        return is_unhappy(locations[i], types[i], i, locations, types, params)

    all_unhappy = vmap(check_agent)(jnp.arange(n))
    # jnp.where with size= returns fixed-length array (required for JIT)
    # Pads with fill_value=-1 when fewer than n agents are unhappy
    indices = jnp.where(all_unhappy, size=n, fill_value=-1)[0]
    count = jnp.sum(all_unhappy)  # number of valid indices
    return indices, count
```

### Parallel Update Functions

```{code-cell} ipython3
@partial(jit, static_argnames=('params',))
def find_happy_candidate(i, locations, types, key, params):
    """
    Propose num_candidates random locations for agent i.
    Return the first one where agent is happy, or current location if none work.
    """
    num_candidates = params.num_candidates
    current_loc = locations[i, :]
    agent_type = types[i]

    keys = random.split(key, num_candidates)
    candidates = vmap(lambda k: random.uniform(k, shape=(2,)))(keys)

    def check_candidate(loc):
        return ~is_unhappy(loc, agent_type, i, locations, types, params)

    happy_at_candidates = vmap(check_candidate)(candidates)

    first_happy_idx = jnp.argmax(happy_at_candidates)
    any_happy = jnp.any(happy_at_candidates)

    new_loc = jnp.where(any_happy, candidates[first_happy_idx], current_loc)
    return new_loc


@partial(jit, static_argnames=('params',))
def parallel_update_step(locations, types, key, params):
    """
    One step of the parallel algorithm:
    1. Generate keys for all agents
    2. For each agent, find a happy candidate location (in parallel)
    3. Only update unhappy agents
    """
    n = params.num_of_type_0 + params.num_of_type_1

    keys = random.split(key, n + 1)
    key = keys[0]
    agent_keys = keys[1:]

    def try_move(i):
        return find_happy_candidate(i, locations, types, agent_keys[i], params)

    new_locations = vmap(try_move)(jnp.arange(n))

    def check_agent(i):
        return is_unhappy(locations[i], types[i], i, locations, types, params)

    is_unhappy_mask = vmap(check_agent)(jnp.arange(n))

    final_locations = jnp.where(is_unhappy_mask[:, None], new_locations, locations)

    return final_locations, key
```

### Type Flipping

This is the key addition. After each iteration, we randomly flip the type of
each agent with probability `flip_prob`.

```{code-cell} ipython3
@partial(jit, static_argnames=('params',))
def flip_types(types, key, params):
    """
    Randomly flip agent types with probability flip_prob.
    """
    n = params.num_of_type_0 + params.num_of_type_1
    flip_prob = params.flip_prob

    # Generate random numbers for each agent
    random_vals = random.uniform(key, shape=(n,))

    # Determine which agents get flipped
    should_flip = random_vals < flip_prob

    # Flip: 0 -> 1, 1 -> 0  (equivalent to 1 - type)
    flipped_types = 1 - types

    # Apply flips only where should_flip is True
    new_types = jnp.where(should_flip, flipped_types, types)

    return new_types
```

## Plotting

```{code-cell} ipython3
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
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)
    locations, types = initialize_state(init_key, params)

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
            _, num_unhappy = get_unhappy_agents(locations, types, params)
            elapsed = time.time() - start_time
            print(f'Iteration {iteration}: {num_unhappy} unhappy agents, {elapsed:.1f}s elapsed')
            plot_distribution(locations, types, f'Iteration {iteration}')

    elapsed = time.time() - start_time
    print(f'\nCompleted {max_iter} iterations in {elapsed:.2f} seconds.')

    return locations, types
```

## Warming Up JAX

```{code-cell} ipython3
key = random.PRNGKey(0)
key, init_key = random.split(key)
test_locations, test_types = initialize_state(init_key, params)

_ = get_distances(test_locations[0], test_locations)
_ = get_neighbors(test_locations[0], 0, test_locations, params)
_ = is_unhappy(test_locations[0], test_types[0], 0, test_locations, test_types, params)
_, _ = get_unhappy_agents(test_locations, test_types, params)

key, subkey = random.split(key)
_ = find_happy_candidate(0, test_locations, test_types, subkey, params)

key, subkey = random.split(key)
_, _ = parallel_update_step(test_locations, test_types, subkey, params)

key, subkey = random.split(key)
_ = flip_types(test_types, subkey, params)

print("JAX functions compiled and ready!")
```

## Results

Let's run the simulation and observe how the system evolves over time.

```{code-cell} ipython3
locations, types = run_simulation_with_shocks(params, max_iter=1000, plot_every=200)
```

## Discussion

The figures show a striking result: segregation levels at the end of the
simulation are much higher than in the basic model without shocks.

Why does this happen? In the basic model, the system converges to an
equilibrium where everyone is happy, and then the dynamics stop. But with
persistent shocks, the system never converges. Each time an agent's type is
flipped, they may suddenly find themselves unhappy (surrounded by agents of
the now-different type). This triggers movement, which can make other agents
unhappy, leading to cascades of relocations.

The key insight is that **the segregation dynamics never shut off**. The same
forces that drove initial segregation in the basic model continue operating
indefinitely:

1. Random type flips create local pockets of unhappiness
2. Unhappy agents relocate to find compatible neighbors
3. This relocation can trigger further unhappiness in other agents
4. The cycle continues, pushing segregation ever higher

This is arguably more realistic than the static equilibrium of the basic model.
Real cities experience constant population turnover—people move in and out,
neighborhoods change. The Schelling dynamics don't just operate once and stop;
they operate continuously on the evolving population.

The persistent shocks prevent the system from settling into equilibrium,
keeping the segregation pressures active. The result is that segregation
continues to increase over time, reaching levels far beyond what we observe
when the system is allowed to converge.
