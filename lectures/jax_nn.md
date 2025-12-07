---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Neural Network Regression with JAX 

```{include} _admonition/gpu.md
```

## Outline

In a {doc}`previous lecture <keras>`, we showed how to implement regression
using a neural network via the deep learning library [Keras](https://keras.io/).

In this lecture, we solve the same problem directly, using JAX operations rather than relying on the Keras frontend.


The objectives are

* Understand the nuts and bolts of the exercise better
* Explore more features of JAX
* Observe how using JAX directly allows us to greatly improve performance.

The lecture proceeds in three stages:

1. We solve the problem using Keras, to give ourselves a benchmark.  
1. We solve the same problem in pure JAX, using pytree operations and gradient descent.  
1. We solve the same problem using a combination of JAX and [Optax](https://optax.readthedocs.io/en/latest/index.html), an optimization library built for JAX.  


We begin with imports and installs.

```{code-cell} ipython3
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from time import time
from typing import NamedTuple
from functools import partial

```

```{code-cell} ipython3
:tags: [hide-output]

!pip install keras optax
```

```{code-cell} ipython3
os.environ['KERAS_BACKEND'] = 'jax'
```

```{note}
Without setting the backend to JAX, the imports below might fail.

If you have problems running the next cell in Jupyter, try 

1. quitting
2. running `export KERAS_BACKEND="jax"` 
3. starting Jupyter on the command line from the same terminal.
```

```{code-cell} ipython3
import keras
from keras import Sequential
from keras.layers import Dense
import optax
```

## Set Up

Here we briefly describe the problem and generate synthetic data.


### Flow

We use the routine from {doc}`keras` to generate data for one-dimensional
nonlinear regression.

Then we will create a dense (i.e., fully connected) neural network with
4 layers, where the input and hidden layers map to $k$-dimensional output space.

The inputs and outputs are scalar (for one-dimensional nonlinear regression), so
the overall mapping is

$$ \mathbb R \to \mathbb R^k \to \mathbb R^k \to \mathbb R^k \to \mathbb R $$

Here's a class to store the learning-related constants we’ll use across all implementations.

Our default value of $k$ will be 10.

```{code-cell} ipython3
class Config(NamedTuple):
    epochs: int = 4000             # Number of passes through the data set
    output_dim: int = 10           # Output dimension of input and hidden layers
    learning_rate: float = 0.001   # Learning rate for gradient descent
    layer_sizes: tuple = (1, 10, 10, 10, 1)  # Sizes of each layer in the network
    seed: int = 14                 # Random seed for data generation
```

### Data

Here's the function to generate the data for our regression analysis.

```{code-cell} ipython3
def generate_data(
        key: jax.Array,         # JAX random key
        data_size: int = 400,   # Sample size
        x_min: float = 0.0,     # Minimum x value
        x_max: float = 5.0      # Maximum x value
    ):
    """
    Generate synthetic regression data.
    """
    x = jnp.linspace(x_min, x_max, num=data_size)
    ϵ = 0.2 * jax.random.normal(key, shape=(data_size,))
    y = x**0.5 + jnp.sin(x) + ϵ
    # Return observations as column vectors
    x = jnp.reshape(x, (data_size, 1))
    y = jnp.reshape(y, (data_size, 1))
    return x, y
```

Here's a plot of the data.

```{code-cell} ipython3
config = Config()
key = jax.random.PRNGKey(config.seed)
key_train, key_validate = jax.random.split(key)
x_train, y_train = generate_data(key_train)
x_validate, y_validate = generate_data(key_validate)
fig, ax = plt.subplots()
ax.scatter(x_train, y_train, alpha=0.5)
ax.scatter(x_validate, y_validate, color='red', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

## Training with Keras

We build a Keras model that can fit a nonlinear function to the generated data
using an ANN.

We will use this fit as a benchmark to test our JAX code.

Since its role is only a benchmark, we refer readers to the {doc}`previous lecture <keras>` for details on the Keras interface.

We start with a function to build the model.

```{code-cell} ipython3
def build_keras_model(
        config: Config,                     # contains configuration data
        activation_function: str = 'tanh'   # activation with default
    ):
    model = Sequential()
    # Add layers to the network sequentially, from inputs towards outputs
    for i in range(len(config.layer_sizes) - 1):
        model.add(
           Dense(units=config.output_dim, activation=activation_function)
        )
    # Add a final layer that maps to a scalar value, for regression.
    model.add(Dense(units=1))
    # Embed training configurations
    model.compile(
        optimizer=keras.optimizers.SGD(),
        loss='mean_squared_error'
    )
    return model
```

Notice that we've set the optimizer to use stochastic gradient descent and a
mean square loss.

Here is a function to train the model.

```{code-cell} ipython3
def train_keras_model(
        model,          # Instance of Keras Sequential model
        x,              # Training data, inputs 
        y,              # Training data, outputs 
        x_validate,     # Validation data, inputs
        y_validate,     # Validation data, outputs
        config: Config  # contains configuration data
    ):
    print(f"Training NN using Keras.")
    start_time = time()
    training_history = model.fit(
        x, y,
        batch_size=max(x.shape),
        verbose=0,
        epochs=config.epochs,
        validation_data=(x_validate, y_validate)
    )
    elapsed = time() - start_time
    mse = model.evaluate(x_validate, y_validate, verbose=2)
    print(f"Trained in {elapsed:.2f} seconds, validation data MSE = {mse}")
    return model, training_history, elapsed, mse
```

The next function extracts and visualizes a prediction from the trained model.

```{code-cell} ipython3
def plot_keras_output(model, x, y, x_validate, y_validate):
    y_predict = model.predict(x_validate, verbose=2)
    fig, ax = plt.subplots()
    ax.scatter(x_validate, y_validate, color='red', alpha=0.5)
    ax.plot(x_validate, y_predict, label="fitted model", color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
```

Let's run the Keras training:

```{code-cell} ipython3
config = Config()
model = build_keras_model(config)
model, training_history, keras_runtime, keras_mse = train_keras_model(
    model, x_train, y_train, x_validate, y_validate, config
)
plot_keras_output(model, x_train, y_train, x_validate, y_validate)
```

The fit is good and we note the relatively low final MSE.


## Training with JAX

For the JAX implementation, we need to construct the network ourselves, as a map
from inputs to outputs.

We’ll use the same network structure we used for the Keras implementation.

### Background and set up

The neural network has the form

$$
f(\theta, x) 
    = (A_3 \circ \sigma \circ A_2 \circ \sigma \circ A_1 \circ \sigma \circ A_0)(x)
$$

Here

- $ x $ is a scalar input – a point on the horizontal axis in the Keras estimation above,  
- $ \circ $ means composition of maps,  
- $ \sigma $ is the activation function – in our case, $ \tanh $, and  
- $ A_i $ represents the affine map $ A_i x = W_i x + b_i $.  

Each matrix $ W_i $ is called a **weight matrix** and each vector $ b_i $ is called a **bias** term.

The symbol $ \theta $ represents the entire collection of parameters:

$$
\theta = (W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3)
$$

In fact, when we implement the affine map $ A_i x = W_i x + b_i $, we will work
with row vectors rather than column vectors, so that

- $ x $ and $ b_i $ are stored as row vectors, and
- the mapping is executed by JAX via the expression `x @ W + b`.

Here's a class to store parameters for one layer of the network.

```{code-cell} ipython3
class LayerParams(NamedTuple):
    """
    Stores parameters for one layer of the neural network.

    """
    W: jnp.ndarray     # weights
    b: jnp.ndarray     # biases
```

The following function initializes a single layer of the network using He
initialization for weights and ones for biases.

```{code-cell} ipython3
def initialize_layer(in_dim, out_dim, key):
    """
    Initialize weights and biases for a single layer of a the network.
    Use He initialization for weights and ones for biases.

    """
    W = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
    b = jnp.ones((1, out_dim))
    return LayerParams(W, b)
```

The next function builds an entire network, as represented by its parameters, by
initializing layers and stacking them into a list.

```{code-cell} ipython3
def initialize_network(
        key: jax.Array,     # JAX random key
        config: Config      # contains configuration data
    ):
    """
    Build a network by initializing all of the parameters.
    A network is a list of LayerParams instances, each
    containing a weight-bias pair (W, b).

    """
    layer_sizes = config.layer_sizes
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        layer = initialize_layer(
            layer_sizes[i],      # in dimension for layer
            layer_sizes[i + 1],  # out dimension for layer
            subkey
        )
        params.append(layer)
    return params
```

Wait, you say!

Shouldn’t we concatenate the elements of $ \theta $ into some kind of big array, so that we can do autodiff with respect to this array?

Actually we don’t need to --- we use the JAX PyTree approach discussed below.


### Coding the network

Here’s our implementation of the ANN $f$:

```{code-cell} ipython3
def f(
        θ: list,                        # Network parameters (pytree)
        x: jnp.ndarray,                 # Input data (row vector)
        σ: callable = jnp.tanh          # Activation function
    ):
    """
    Perform a forward pass over the network to evaluate f(θ, x).
    """
    *hidden, last = θ
    for layer in hidden:
        x = σ(x @ layer.W + layer.b)
    x = x @ last.W + last.b
    return x 
```

The function $ f $ is appropriately vectorized, so that we can pass in the entire
set of input observations as `x` and return the predicted vector of outputs `y_hat = f(θ, x)`
corresponding  to each data point.

The loss function is mean squared error, the same as the Keras case.

```{code-cell} ipython3
def loss_fn(
        θ: list,            # Network parameters (pytree)
        x: jnp.ndarray,     # Input data
        y: jnp.ndarray      # Target data
    ):
    return jnp.mean((f(θ, x) - y)**2)
```

We’ll use its gradient to do stochastic gradient descent.

(Technically, we will be doing gradient descent, rather than stochastic
gradient descent, since will not randomize over sample points when we
evaluate the gradient.)

```{code-cell} ipython3
loss_gradient = jax.jit(jax.grad(loss_fn))
```

The gradient of `loss_fn` is with respect to the first argument `θ`.

The code above seems kind of magical, since we are differentiating with respect
to a parameter “vector” stored as a list of dictionaries containing arrays.

How can we differentiate with respect to such a complex object?

The answer is that the list of dictionaries is treated internally as a
[pytree](https://docs.jax.dev/en/latest/pytrees.html).

The JAX function `grad` understands how to

1. extract the individual arrays (the "leaves" of the tree),
1. compute derivatives with respect to each one, and
1. pack the resulting derivatives into a pytree with the same structure as the parameter vector.

+++

### Gradient descent

Using the above code, we can now write our rule for updating the parameters via gradient descent, which is the
algorithm we covered in our [lecture on autodiff](https://jax.quantecon.org/autodiff.html).

In this case, to keep things as simple as possible, we’ll use a fixed learning rate for every iteration.

```{code-cell} ipython3
def update_parameters(
        θ: list,            # Current parameters (pytree)
        x: jnp.ndarray,     # Input data
        y: jnp.ndarray,     # Target data
        config: Config      # contains configuration data
    ):
    """
    Update the parameter pytree using gradient descent.

    """
    λ = config.learning_rate
    # Specify the update rule
    def gradient_descent_step(p, g):
        """
        A rule for updating parameter vector p given gradient vector g.
        It will be applied to each leaf of the pytree of parameters.
        """
        return p - λ * g
    gradient = loss_gradient(θ, x, y)
    # Use tree.map to apply the update rule to the parameter vectors
    θ_new = jax.tree.map(gradient_descent_step, θ, gradient)
    return θ_new
```

Here `jax.tree.map` understands `θ` and `gradient` as pytrees of the
same structure and executes `p - λ * g` on the corresponding leaves of the pair
of trees.

Each weight matrix and bias vector is updated by gradient
descent, exactly as required.

Here’s code that puts this all together.

```{code-cell} ipython3
@partial(jax.jit, static_argnames=['config'])
def train_jax_model(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        x_validate: jnp.ndarray,    # Validation input data
        y_validate: jnp.ndarray,    # Validation target data
        config: Config              # contains configuration data
    ):
    """
    Train model using gradient descent.

    """
    def update(_, θ):
        θ_new = update_parameters(θ, x, y, config)
        return θ_new

    θ_final = jax.lax.fori_loop(0, config.epochs, update, θ)
    return θ_final
```

### Execution

Let's run our code and see how it goes.

We'll reuse the data we generated earlier.

```{code-cell} ipython3
# Reset parameter vector
config = Config()
param_key = jax.random.PRNGKey(1234)
θ = initialize_network(param_key, config)

# Warmup run to trigger JIT compilation
train_jax_model(θ, x_train, y_train, x_validate, y_validate, config)

# Reset and time the actual run
θ = initialize_network(param_key, config)
start_time = time()
θ = train_jax_model(θ, x_train, y_train, x_validate, y_validate, config)
θ[0].W.block_until_ready()  # Ensure computation completes
jax_runtime = time() - start_time

jax_mse = loss_fn(θ, x_validate, y_validate)
jax_train_mse = loss_fn(θ, x_train, y_train)
print(f"Trained model with JAX in {jax_runtime:.2f} seconds.")
print(f"Final MSE on validation data = {jax_mse:.6f}")
```

Despite the simplicity of our implementation, we actually perform slightly better than Keras.

Here's a visualization of the quality of our fit.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x_validate, y_validate, color='red', alpha=0.5)
ax.plot(x_validate.flatten(), f(θ, x_validate).flatten(),
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

## JAX plus Optax

Our hand-coded optimization routine above was quite effective, but in practice
we might wish to use an optimization library written for JAX.

One such library is [Optax](https://optax.readthedocs.io/en/latest/).

### Optax with SGD

Here’s a training routine using Optax’s stochastic gradient descent solver.

```{code-cell} ipython3
@partial(jax.jit, static_argnames=['config'])
def train_jax_optax(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        config: Config              # contains configuration data
    ):
    " Train model using Optax SGD optimizer. "
    epochs = config.epochs
    learning_rate = config.learning_rate
    solver = optax.sgd(learning_rate)
    opt_state = solver.init(θ)

    def update(_, loop_state):
        θ, opt_state = loop_state
        grad = loss_gradient(θ, x, y)
        updates, new_opt_state = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        new_loop_state = θ_new, new_opt_state
        return new_loop_state

    initial_loop_state = θ, opt_state
    final_loop_state = jax.lax.fori_loop(0, epochs, update, initial_loop_state)
    θ_final, _ = final_loop_state
    return θ_final
```

Let’s try running it.

```{code-cell} ipython3
# Reset parameter vector
θ = initialize_network(param_key, config)

# Warmup run to trigger JIT compilation
train_jax_optax(θ, x_train, y_train, config)

# Reset and time the actual run
θ = initialize_network(param_key, config)
start_time = time()
θ = train_jax_optax(θ, x_train, y_train, config)
θ[0].W.block_until_ready()  # Ensure computation completes
optax_sgd_runtime = time() - start_time

optax_sgd_mse = loss_fn(θ, x_validate, y_validate)
optax_sgd_train_mse = loss_fn(θ, x_train, y_train)
print(f"Trained model with JAX and Optax SGD in {optax_sgd_runtime:.2f} seconds.")
print(f"Final MSE on validation data = {optax_sgd_mse:.6f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x_validate, y_validate, color='red', alpha=0.5)
ax.plot(x_validate.flatten(), f(θ, x_validate).flatten(),
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

### Optax with ADAM

We can also consider using a slightly more sophisticated gradient-based method,
such as [ADAM](https://arxiv.org/pdf/1412.6980).

You will notice that the syntax for using this alternative optimizer is very
similar.

```{code-cell} ipython3
@partial(jax.jit, static_argnames=['config'])
def train_jax_optax_adam(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        config: Config              # contains configuration data
    ):
    " Train model using Optax ADAM optimizer. "
    epochs = config.epochs
    learning_rate = config.learning_rate
    solver = optax.adam(learning_rate)
    opt_state = solver.init(θ)

    def update(_, loop_state):
        θ, opt_state = loop_state
        grad = loss_gradient(θ, x, y)
        updates, new_opt_state = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        return (θ_new, new_opt_state)

    initial_loop_state = θ, opt_state
    θ_final, _ = jax.lax.fori_loop(0, epochs, update, initial_loop_state)
    return θ_final
```


```{code-cell} ipython3
# Reset parameter vector
θ = initialize_network(param_key, config)

# Warmup run to trigger JIT compilation
train_jax_optax_adam(θ, x_train, y_train, config)

# Reset and time the actual run
θ = initialize_network(param_key, config)
start_time = time()
θ = train_jax_optax_adam(θ, x_train, y_train, config)
θ[0].W.block_until_ready()  # Ensure computation completes
optax_adam_runtime = time() - start_time

optax_adam_mse = loss_fn(θ, x_validate, y_validate)
optax_adam_train_mse = loss_fn(θ, x_train, y_train)
print(f"Trained model with JAX and Optax ADAM in {optax_adam_runtime:.2f} seconds.")
print(f"Final MSE on validation data = {optax_adam_mse:.6f}")
```

Here's a visualization of the result.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x_validate, y_validate, color='red', alpha=0.5)
ax.plot(x_validate.flatten(), f(θ, x_validate).flatten(),
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

## Summary

Here we compare the performance of the four different training approaches we explored in this lecture.

```{code-cell} ipython3
:tags: [hide-input]

import pandas as pd

# Compute training MSEs for each method
# Need to retrieve the trained models and compute training MSE
# For Keras, we already have the model
keras_train_mse = model.evaluate(x_train, y_train, verbose=0)

# For JAX methods, we need to compute using loss_fn with the final θ from each method
# We need to re-train or save the θ from each method
# For now, let's add these calculations after each training section

# Create summary table
results = {
    'Method': [
        'Keras',
        'Pure JAX (hand-coded GD)',
        'JAX + Optax SGD',
        'JAX + Optax ADAM'
    ],
    'Runtime (s)': [
        keras_runtime,
        jax_runtime,
        optax_sgd_runtime,
        optax_adam_runtime
    ],
    'Training MSE': [
        keras_train_mse,
        jax_train_mse,
        optax_sgd_train_mse,
        optax_adam_train_mse
    ],
    'Validation MSE': [
        keras_mse,
        jax_mse,
        optax_sgd_mse,
        optax_adam_mse
    ]
}

df = pd.DataFrame(results)
# Format MSE columns to 6 decimal places
df['Training MSE'] = df['Training MSE'].apply(lambda x: f"{x:.6f}")
df['Validation MSE'] = df['Validation MSE'].apply(lambda x: f"{x:.6f}")
print("\nSummary of Training Methods:")
print(df.to_string(index=False))
```


All methods achieve similar validation MSE values (around 0.043-0.045).

At the time of writing, the MSEs from plain vanilla Optax and our own hand-coded SGD routine are identical.

The ADAM optimizer achieves slightly better MSE by using adaptive learning rates.

Still, our hand-coded algorithm does very well compared to this high-quality optimizer.

Note also that the pure JAX implementations are significantly faster than Keras.

This is because JAX can JIT-compile the entire training loop.

Not surprisingly, Keras has more overhead from its abstraction layers.


## Exercises

```{exercise}
:label: jax_nn_ex1

Try to reduce the MSE on the validation data without significantly increasing
the computational load.

You should hold constant both the number of epochs and the total number of
parameters in the network.

Currently, the network has 4 layers with output dimension $k=10$, giving a total
of $251$ parameters

You can experiment with:
- Changing the network architecture 
- Trying different activation functions 
- Modifying the optimizer (e.g., learning rates, learning rate schedules, momentum, etc.)
- Experimenting with different weight initialization strategies
- Modifying the loss function (e.g., adding regularization)


Which combination gives you the lowest validation MSE?
```


```{solution-start} jax_nn_ex1
:class: dropdown
```

Let's implement and test several strategies. 

**Strategy 1: Deeper Network + LR Schedule + L2 Regularization**

Let's try a deeper network (6 layers) combined with learning rate schedule and L2 regularization:

```{code-cell} ipython3
# Strategy 1: Deeper network + LR schedule + L2 regularization
# Define deeper network architecture
def initialize_deep_params(
        key: jax.Array,
        k: int = 6,
        num_hidden: int = 5
    ):
    " Initialize parameters for deeper network with k=6. "
    layer_sizes = tuple([1] + [k] * num_hidden + [1])
    config_deep = Config(layer_sizes=layer_sizes)
    return initialize_network(key, config_deep)

config_deep = Config(layer_sizes=(1, 6, 6, 6, 6, 6, 1))
θ_deep = initialize_deep_params(param_key)

def train_deep_with_schedule_and_l2(
        θ: list,
        x: jnp.ndarray,
        y: jnp.ndarray,
        config: Config,
        lambda_l2: float = 0.001
    ):
    " Train deeper network with learning rate schedule and L2 regularization. "
    epochs = config.epochs
    schedule = optax.exponential_decay(
        init_value=0.003,
        transition_steps=1000,
        decay_rate=0.5
    )

    # Define regularized loss function
    @jax.jit
    def loss_fn_l2(θ, x, y):
        # Standard MSE loss
        mse = jnp.mean((f(θ, x) - y)**2)
        # L2 penalty on weights (not biases)
        l2_penalty = 0.0
        for W, b in θ:
            l2_penalty += jnp.sum(W**2)
        return mse + lambda_l2 * l2_penalty

    loss_gradient_l2 = jax.jit(jax.grad(loss_fn_l2))

    solver = optax.adam(schedule)
    opt_state = solver.init(θ)

    def update(_, loop_state):
        θ, opt_state = loop_state
        grad = loss_gradient_l2(θ, x, y)
        updates, new_opt_state = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        return (θ_new, new_opt_state)

    initial_loop_state = θ, opt_state
    θ_final, _ = jax.lax.fori_loop(0, epochs, update, initial_loop_state)
    return θ_final

# Warmup
train_deep_with_schedule_and_l2(θ_deep, x_train, y_train, config_deep)

# Actual run
θ_deep = initialize_deep_params(param_key)
start_time = time()
θ_deep_l2 = train_deep_with_schedule_and_l2(θ_deep, x_train, y_train, config_deep)
θ_deep_l2[0].W.block_until_ready()
deep_l2_runtime = time() - start_time

deep_l2_mse = loss_fn(θ_deep_l2, x_validate, y_validate)
print(f"Strategy 1 - Deeper network + LR schedule + L2 regularization")
print(f"  Runtime: {deep_l2_runtime:.2f}s")
print(f"  Validation MSE: {deep_l2_mse:.6f}")
print(f"  Improvement over ADAM: {optax_adam_mse - deep_l2_mse:.6f}")
```

**Strategy 2: Baseline + Armijo Line Search**

Let's implement gradient descent with [Armijo line search](https://en.wikipedia.org/wiki/Backtracking_line_search) for adaptive step size selection:

```{code-cell} ipython3
# Strategy 2: Baseline architecture + Armijo line search
# Line search parameters
line_search_init_value = 0.01
line_search_backtrack_factor = 0.5
line_search_armijo_constant = 0.001
max_backtrack_steps = 20

@partial(jax.jit, static_argnames=['config'])
def train_jax_armijo_ls(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        config: Config              # contains configuration data
    ):
    """
    Train model using gradient descent with Armijo line search.

    The Armijo line search adaptively finds a suitable step size at each
    iteration by ensuring sufficient decrease in the loss function.
    """
    epochs = config.epochs

    # Line search parameters
    init_alpha = line_search_init_value
    backtrack_factor = line_search_backtrack_factor
    _armijo_constant = line_search_armijo_constant

    def update_step(current_theta, x_data, y_data):
        current_loss = loss_fn(current_theta, x_data, y_data)
        grad = loss_gradient(current_theta, x_data, y_data)

        # Calculate squared Euclidean norm of the gradient for Armijo condition
        grad_norm_sq = jax.tree_util.tree_reduce(
            lambda a, b: a + jnp.sum(b**2), grad, initializer=0.0
        )

        # Define the condition for the while_loop
        def cond_fn(loop_args):
            alpha_val, current_loss_val, grad_sq_sum, theta_orig, x_in, y_in, step_count = loop_args
            loss_threshold = current_loss_val - _armijo_constant * alpha_val * grad_sq_sum
            theta_candidate = jax.tree.map(lambda p, g_leaf: p - alpha_val * g_leaf, theta_orig, grad)
            loss_candidate = loss_fn(theta_candidate, x_in, y_in)
            return (loss_candidate > loss_threshold) & (step_count < max_backtrack_steps)

        # Define the body for the while_loop
        def body_fn(loop_args):
            alpha_val, current_loss_val, grad_sq_sum, theta_orig, x_in, y_in, step_count = loop_args
            new_alpha = alpha_val * backtrack_factor
            new_step_count = step_count + 1
            return (new_alpha, current_loss_val, grad_sq_sum, theta_orig, x_in, y_in, new_step_count)

        # Execute the Armijo line search using jax.lax.while_loop
        final_alpha, _, _, _, _, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (init_alpha, current_loss, grad_norm_sq, current_theta, x_data, y_data, 0)
        )

        # Update parameters with the chosen step size
        theta_new = jax.tree.map(lambda p, g_leaf: p - final_alpha * g_leaf, current_theta, grad)
        return theta_new

    # Main training loop (epochs)
    θ_final = jax.lax.fori_loop(0, epochs, lambda i, current_theta: update_step(current_theta, x, y), θ)
    return θ_final

# Warmup
θ = initialize_network(param_key, config)
train_jax_armijo_ls(θ, x_train, y_train, config)

# Actual run
θ = initialize_network(param_key, config)
start_time = time()
θ_armijo = train_jax_armijo_ls(θ, x_train, y_train, config)
θ_armijo[0].W.block_until_ready()
armijo_runtime = time() - start_time

armijo_mse = loss_fn(θ_armijo, x_validate, y_validate)
print(f"Strategy 2 - Baseline + Armijo Line Search")
print(f"  Runtime: {armijo_runtime:.2f}s")
print(f"  Validation MSE: {armijo_mse:.6f}")
print(f"  Improvement over ADAM: {optax_adam_mse - armijo_mse:.6f}")
```

**Results Summary**

Let's compare all strategies:

```{code-cell} ipython3
:tags: [hide-input]

# Summary of all strategies
strategies_results = {
    'Strategy': [
        'Baseline (ADAM + tanh)',
        '1. Deeper network + LR schedule + L2',
        '2. Baseline + Armijo Line Search'
    ],
    'Runtime (s)': [
        optax_adam_runtime,
        deep_l2_runtime,
        armijo_runtime
    ],
    'Validation MSE': [
        optax_adam_mse,
        deep_l2_mse,
        armijo_mse
    ],
    'Improvement': [
        0.0,
        float(optax_adam_mse - deep_l2_mse),
        float(optax_adam_mse - armijo_mse)
    ]
}

df_strategies = pd.DataFrame(strategies_results)
print("\nSummary of Exercise Strategies:")
print(df_strategies.to_string(index=False))
```


In terms of reducing loss on the validation test data, the current winner is the
Armijo line search strategy. 

The Armijo backtracking line search is an adaptive step size method that
dynamically adjusts the learning rate at each iteration to ensure sufficient
decrease in the loss function. 

Unlike fixed learning rates or predetermined schedules, it adapts to the local
geometry of the loss landscape.

This strategy and its code was contributed by [Matyas Farkas](https://www.matyasfarkas.eu/).



```{solution-end}
```

