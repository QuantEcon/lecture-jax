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


# Schelling's Model of Racial Segregation 

## Outline

Racial residential segregation has significant consequences across many
dimensions of life in the US and other countries.

In 1969, Thomas C. Schelling developed a simple but striking model of racial
segregation ([Schelling, 1969](https://en.wikipedia.org/wiki/Schelling%27s_model_of_segregation)).

His model studies the dynamics of racially mixed neighborhoods.

Like much of Schelling's work, the model shows how local interactions can lead
to surprising aggregate outcomes.

It studies a setting where agents (think of households) have relatively mild
preference for neighbors of the same race.

For example, these agents might be comfortable with a mixed race neighborhood
but uncomfortable when they feel "surrounded" by people from a different race.

Schelling illustrated the following surprising result: in such a setting, mixed
race neighborhoods are likely to be unstable, tending to collapse over time.

In fact the model predicts strongly divided neighborhoods, with high levels of
segregation.

In other words, extreme segregation outcomes arise even though people's
preferences are not particularly extreme.

These extreme outcomes happen because of *interactions* between agents in the
model (e.g., households in a city) that drive self-reinforcing dynamics in the
model.

In recognition of his work on segregation and other research, Schelling was
awarded the 2005 Nobel Prize in Economic Sciences.

In this and the following lectures, we study Schelling's model using simulation.

We will start in this lecture with a relatively elementary version written in
pure Python.

Once that simple version is in place, we will consider efficiency.

We will develop progressively more efficient versions of the model, allowing us
to study dynamics with larger populations and more interesting features.

In the most efficient versions, JAX and parallelization will play a central
role.

Let's start with some imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from random import uniform, seed
from math import sqrt
import time
```

## Background

Before jumping into the modelling process, let's look at some data

The maps below are from the
[Weldon Cooper Center for Public Service](https://demographics.coopercenter.org/)
at the University of Virginia.

They illustrate the racial composition of several major US cities, based on census data. 

Each dot represents a group of residents --- we omit details on which color represents which group.

These maps reveal patterns of spatial separation between racial groups.

### Columbus, Ohio

```{figure} _static/fig/columbus.webp
:name: columbus_map
:width: 80%

Racial distribution in Columbus, Ohio. 
```

### Memphis, Tennessee

```{figure} _static/fig/memphis.webp
:name: memphis_map
:width: 80%

Racial distribution in Memphis, Tennessee. 
```

### Washington, D.C.

```{figure} _static/fig/washington_dc.webp
:name: dc_map
:width: 80%

Racial distribution in Washington, D.C. 
```

### Houston, Texas

```{figure} _static/fig/houston.webp
:name: houston_map
:width: 80%

Racial distribution in Houston, Texas.
```

### Miami, Florida

```{figure} _static/fig/miami.webp
:name: miami_map
:width: 80%

Racial distribution in Miami, Florida. 
```

Looking at these maps, one might assume that segregation persists because of
strong preferences---that people simply want to live only with others of
their own race.

But is this actually the case?

Let's now look at Schelling's segregation model,
which demonstrates a surprising result: extreme segregation can emerge even
when individuals have only mild preferences for same-race neighbors.

This insight has profound implications for understanding how segregation
persists and what policies might effectively address it.


## The model

### Set-Up

We will cover a variation of Schelling's model that is different from the
original but also easy to program and, at the same time, captures his main
idea.

For now our coding objective is clarity rather than efficiency.

Suppose we have two types of people: orange people and green people.

Assume there are $n$ of each type.

These agents all live on a single unit square.

Thus, the location (e.g, address) of an agent is just a point $(x, y)$,  where
$0 < x, y < 1$.

* The set of all points $(x,y)$ satisfying $0 < x, y < 1$ is called the **unit square**
* Below we denote the unit square by $S$

+++

### Preferences

We will say that an agent is 

* **happy** if $k \leq 6$ of her 10 nearest neighbors are of a different type.
* **unhappy** if $k > 6$ of her 10 nearest neighbors are of a different type.

For example,

*  if an agent is orange and 6 of her 10 nearest neighbors are green, then she is happy.
*  if an agent is orange and 7 of her 10 nearest neighbors are green, then she is unhappy.

'Nearest' is in terms of [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance).

An important point to note is that agents are **not** averse to living in mixed areas.

They are perfectly happy if 60% of their neighbors are of the other color.

Let's set up the parameters for our simulation:

```{code-cell} ipython3
seed(1234)              # set seed for reproducibility

num_of_type_0 = 1000    # number of agents of type 0 (orange)
num_of_type_1 = 1000    # number of agents of type 1 (green)
num_neighbors = 10      # number of agents viewed as neighbors
max_other_type = 6      # max number of different-type neighbors tolerated
```

+++

### Behavior

Initially, agents are mixed together (integrated).

In particular, we assume that the initial location of each agent is an
independent draw from a bivariate uniform distribution on the unit square $S$.

* Their $x$ coordinate is drawn from a uniform distribution on $(0,1)$
* Their $y$ coordinate is drawn independently from the same distribution.

Now, cycling through the set of all agents, each agent is now given the chance to stay or move.

Each agent stays if they are happy and moves if they are unhappy.

The algorithm for moving is as follows

```{prf:algorithm} Relocation Algorithm
:label: move_algo

1. Draw a random location in $S$
2. If happy at new location, move there
3. Else go to step 1

```

We cycle continuously through the agents, each time allowing an unhappy agent
to move.

We continue to cycle until no one wishes to move.

+++

## Main Loop

Let's write the code to run this simulation.

In what follows, agents are modeled as [objects](https://python-programming.quantecon.org/python_oop.html) that store


```{code-block} none
    * type (green or orange)
    * location
```

Here's a class that we can use to instantiate agents from:

```{code-cell} ipython3
class Agent:

    def __init__(self, type):
        self.type = type
        self.location = uniform(0, 1), uniform(0, 1)
```


Here's a collection of functions that act on agents:

```{code-cell} ipython3
def get_distance(agent, other_agent):
    "Computes the Euclidean distance between self and other agent."
    a = agent.location[0] - other_agent.location[0]
    b = agent.location[1] - other_agent.location[1]
    return sqrt(a**2 + b**2)


def is_happy(agent, all_agents):
    """
    True if agent has at most max_other_type different-type agents
    among its num_neighbors nearest neighbors.
    """
    # Collect all other agents, sorted by distance to agent
    others = [a for a in all_agents if a != agent]
    others.sort(key=lambda a: get_distance(agent, a))

    # Check the nearest neighbors
    neighbors = others[:num_neighbors]
    num_other_type = sum(neighbor.type != agent.type for neighbor in neighbors)
    return num_other_type <= max_other_type


def move_agent(agent, all_agents, max_attempts=10_000):
    "If not happy, then randomly choose new locations until happy."
    attempts = 0
    while not is_happy(agent, all_agents):
        agent.location = uniform(0, 1), uniform(0, 1)
        attempts += 1
        if attempts >= max_attempts:
            break
```

Here's some code that takes a list of agents and produces a plot showing their
locations on the unit square.

Orange agents are represented by orange dots and green ones are represented by
green dots.

```{code-cell} ipython3
:tags: [hide-input]

def plot_distribution(agents, round_num):
    "Plot the distribution of agents after round_num rounds of the loop."
    x_values_0, y_values_0 = [], []
    x_values_1, y_values_1 = [], []
    # == Obtain locations of each type == #
    for agent in agents:
        x, y = agent.location
        if agent.type == 0:
            x_values_0.append(x)
            y_values_0.append(y)
        else:
            x_values_1.append(x)
            y_values_1.append(y)
    fig, ax = plt.subplots()
    plot_args = {
        'markersize': 6,
        'alpha': 0.8,
        'markeredgecolor': 'black',
        'markeredgewidth': 0.5
    }
    ax.plot(x_values_0, y_values_0,
        'o', markerfacecolor='darkorange', **plot_args)
    ax.plot(x_values_1, y_values_1,
        'o', markerfacecolor='green', **plot_args)
    ax.set_title(f'Round {round_num}')
    plt.show()
```

The main loop cycles through all agents until no one wishes to move.

```{prf:algorithm} Main Simulation Loop
:label: main_loop_algo

**Input:** Set of agents with initial random locations

**Output:** Final distribution of agents

1. Set `count` $\leftarrow$ 1
2. While `count` < `max_iter`:
    1. Set `number_of_moves` $\leftarrow$ 0
    2. For each agent:
        1. If agent is unhappy, relocate using {prf:ref}`move_algo` and increment `number_of_moves`
    3. Plot distribution
    4. Increment `count`
    5. If `number_of_moves` = 0, exit loop

```

The code is below

```{code-cell} ipython3
def run_simulation(all_agents, max_iter=100_000):

    # Initialize a counter
    count = 1

    # Loop until no agent wishes to move
    start_time = time.time()
    while count < max_iter:
        number_of_moves = 0
        for agent in all_agents:
            if not is_happy(agent, all_agents):
                move_agent(agent, all_agents)
                number_of_moves += 1
        # Plot the distribution after this round
        plot_distribution(all_agents, count)
        # Print outcome and stop loop if no one moved
        print(f'Completed loop {count} with {number_of_moves} moves')
        count += 1
        if number_of_moves == 0:
            break
    elapsed = time.time() - start_time

    if count < max_iter:
        print(f'Converged in {elapsed:.2f} seconds after {count} iterations.')
    else:
        print('Hit iteration bound and terminated.')

```


## Results

We are now ready to run our simulation.

First we build a population of agents:

```{code-cell} ipython3
all_agents = []
for i in range(num_of_type_0):
    all_agents.append(Agent(0))
for i in range(num_of_type_1):
    all_agents.append(Agent(1))

plot_distribution(all_agents, 0)
```

Now we run the simulation and look at the results.

```{code-cell} ipython3
run_simulation(all_agents)
```

As discussed above, agents are initially mixed randomly together.

But after several cycles, they become segregated into distinct regions.

In this instance, the program terminated after a small number of cycles
through the set of agents, indicating that all agents had reached a state of
happiness.

We notice that the fully mixed environment rapidly breaks down.

We get emergence of at least some segregation.

This is despite the fact that people in the model don't actually mind living
mixed with the other type.

Even with these preferences, the outcome is some degree of segregation.

(Not a lot of segregation, but we'll see more segregation in later lectures,
after some modifications.)


## Performance

Our Python code was written for readability, not speed.

This is fine for very small simulations but not for bigger ones.

That's a problem for us because we want to experiment with some more ideas.

In the following lectures we'll look at strategies for making our code faster.

Then we'll investigate variations that might lead to even more segregation.
