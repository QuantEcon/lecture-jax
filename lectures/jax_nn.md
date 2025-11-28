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

In a {doc}`previous lecture <keras>`, we showed how to implement regression using a neural network via the popular deep learning library [Keras](https://keras.io/).

In this lecture, we solve the same problem directly, using JAX operations rather than relying on the Keras frontend.

The objective is to understand the nuts and bolts of the exercise better, as
well as to explore more features of JAX.

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

```{code-cell} ipython3
from typing import NamedTuple

class Config(NamedTuple):
    epochs: int = 4000             # Number of passes through the data set
    data_size: int = 400           # Sample size
    num_layers: int = 4            # Depth of the network
    output_dim: int = 10           # Output dimension k of input and hidden layers
    learning_rate: float = 0.001   # Learning rate for gradient descent
```

### Data

Here's the function to generate the data for our regression analysis.

```{code-cell} ipython3
def generate_data(
        key: jax.Array,         # JAX random key
        config: Config,         # contains configuration data
        x_min: float = 0.0,     # Minimum x value
        x_max: float = 5.0      # Maximum x value
    ):
    """
    Generate synthetic regression data.
    Pure functional version using JAX random keys.
    """
    x = jnp.linspace(x_min, x_max, num=config.data_size)
    ϵ = 0.2 * jax.random.normal(key, shape=(config.data_size,))
    y = x**0.5 + jnp.sin(x) + ϵ
    # Return observations as column vectors
    x = jnp.reshape(x, (config.data_size, 1))
    y = jnp.reshape(y, (config.data_size, 1))
    return x, y
```

Here's a plot of the data.

```{code-cell} ipython3
config = Config()
key = jax.random.PRNGKey(1234)
x, y = generate_data(key, config)
fig, ax = plt.subplots()
ax.scatter(x, y)
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
    for i in range(config.num_layers-1):
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
    ax.scatter(x, y)
    ax.plot(x, y_predict, label="fitted model", color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
```

Let's run the Keras training:

```{code-cell} ipython3
config = Config()
model = build_keras_model(config)
key = jax.random.PRNGKey(1234)
key, subkey1, subkey2 = jax.random.split(key, 3)
x, y = generate_data(subkey1, config)
x_validate, y_validate = generate_data(subkey2, config)
model, training_history, keras_runtime, keras_mse = train_keras_model(
    model, x, y, x_validate, y_validate, config
)
plot_keras_output(model, x, y, x_validate, y_validate)
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

Here’s a function to initialize parameters.

The parameter “vector” `θ`  will be stored as a list of dicts.

```{code-cell} ipython3
def initialize_params(
        key: jax.Array,     # JAX random key
        config: Config      # contains configuration data
    ):
    """
    Generate an initial parameterization for a feed forward neural network.
    Pure functional version using JAX random keys.
    """
    k = config.output_dim
    shapes = (
        (1, k),  # W_0.shape
        (k, k),  # W_1.shape
        (k, k),  # W_2.shape
        (k, 1)   # W_3.shape
    )
    # A function to generate weight matrices using JAX random
    def w_init(key, m, n):
        return jax.random.normal(key, shape=(m, n)) * jnp.sqrt(2 / m)
    # Build list of dicts, each containing a (weight, bias) pair
    θ = []
    for w_shape in shapes:
        m, n = w_shape
        key, subkey = jax.random.split(key)
        layer_params = dict(W=w_init(subkey, m, n), b=jnp.ones((1, n)))
        θ.append(layer_params)
    return θ
```

Wait, you say!

Shouldn’t we concatenate the elements of $ \theta $ into some kind of big array, so that we can do autodiff with respect to this array?

Actually we don’t need to --- we use the JAX PyTree approach discussed below.


### Coding the network

Here’s our implementation of the ANN $f$:

```{code-cell} ipython3
@jax.jit
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
        W, b = layer['W'], layer['b']
        x = σ(x @ W + b)
    W, b = last['W'], last['b']
    x = x @ W + b
    return x 
```

The function $ f $ is appropriately vectorized, so that we can pass in the entire
set of input observations as `x` and return the predicted vector of outputs `y_hat = f(θ, x)`
corresponding  to each data point.

The loss function is mean squared error, the same as the Keras case.

```{code-cell} ipython3
@jax.jit
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
@jax.jit
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
    def update(θ, _):
        # Record losses
        train_loss = loss_fn(θ, x, y)
        val_loss = loss_fn(θ, x_validate, y_validate)
        # Update parameters
        θ_new = update_parameters(θ, x, y, config)
        return θ_new, (train_loss, val_loss)

    # Initialize with empty arrays
    θ_final, (training_losses, validation_losses) = jax.lax.scan(
        update, θ, None, length=config.epochs
    )
    return θ_final, training_losses, validation_losses
```

### Execution

Let’s run our code and see how it goes.

We'll reuse the data we generated for the Keras experiment.

```{code-cell} ipython3
config = Config()
param_key = jax.random.PRNGKey(1234)
θ = initialize_params(param_key, config)
```

```{code-cell} ipython3
# Warmup run to trigger JIT compilation
θ_warmup, _, _ = train_jax_model(θ, x, y, x_validate, y_validate, config)

# Reset and time the actual run
θ = initialize_params(param_key, config)
start_time = time()
θ, training_loss, validation_loss = train_jax_model(
    θ, x, y, x_validate, y_validate, config
)
θ[0]['W'].block_until_ready()  # Ensure computation completes
jax_runtime = time() - start_time

jax_mse = loss_fn(θ, x_validate, y_validate)
print(f"Trained model with JAX in {jax_runtime:.2f} seconds.")
print(f"Final MSE on validation data = {jax_mse:.6f}")
```

This figure shows MSE across iterations:

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(range(len(validation_loss)), validation_loss, label='validation loss')
ax.legend()
plt.show()
```

Let’s check the final MSE on the validation data, at the estimated parameters.

```{code-cell} ipython3
print(f"""
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
```

This MSE is not as low as we got for Keras, but we did quite well given how
simple our implementation is.

Here’s a visualization of the quality of our fit.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(), 
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

## JAX plus Optax

Our hand-coded optimization routine above was quite effective, but in practice
we might wish to use an optimization library written for JAX.

One such library is [Optax](https://optax.readthedocs.io/en/latest/).

+++

### Optax with SGD

Here’s a training routine using Optax’s stochastic gradient descent solver.

```{code-cell} ipython3
def train_jax_optax(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        epochs: int = 4000,         # Number of training epochs
        learning_rate: float = 0.001  # Learning rate for optimizer
    ):
    """
    Train model using Optax SGD optimizer.
    Pure functional version using jax.lax.scan.
    """
    solver = optax.sgd(learning_rate)
    opt_state = solver.init(θ)

    def train_step(carry, _):
        θ, opt_state = carry
        grad = loss_gradient(θ, x, y)
        updates, opt_state_new = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        return (θ_new, opt_state_new), None

    (θ_final, _), _ = jax.lax.scan(train_step, (θ, opt_state), None, length=epochs)
    return θ_final
```

Let’s try running it.

```{code-cell} ipython3
# Reset parameter vector
θ = initialize_params(param_key, config)

# Warmup run to trigger JIT compilation
θ_warmup = train_jax_optax(θ, x, y)

# Reset and time the actual run
θ = initialize_params(param_key, config)
start_time = time()
θ = train_jax_optax(θ, x, y)
θ[0]['W'].block_until_ready()  # Ensure computation completes
optax_sgd_runtime = time() - start_time

optax_sgd_mse = loss_fn(θ, x_validate, y_validate)
print(f"Trained model with JAX and Optax SGD in {optax_sgd_runtime:.2f} seconds.")
print(f"Final MSE on validation data = {optax_sgd_mse:.6f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(), 
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
def train_jax_optax_adam(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        epochs: int = 4000,         # Number of training epochs
        learning_rate: float = 0.001  # Learning rate for optimizer
    ):
    """
    Train model using Optax ADAM optimizer.
    Pure functional version using jax.lax.scan.
    """
    solver = optax.adam(learning_rate)
    opt_state = solver.init(θ)

    def train_step(carry, _):
        θ, opt_state = carry
        grad = loss_gradient(θ, x, y)
        updates, opt_state_new = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        return (θ_new, opt_state_new), None

    (θ_final, _), _ = jax.lax.scan(train_step, (θ, opt_state), None, length=epochs)
    return θ_final
```

```{code-cell} ipython3
# Reset parameter vector
θ = initialize_params(param_key, config)

# Warmup run to trigger JIT compilation
θ_warmup = train_jax_optax_adam(θ, x, y)

# Reset and time the actual run
θ = initialize_params(param_key, config)
start_time = time()
θ = train_jax_optax_adam(θ, x, y)
θ[0]['W'].block_until_ready()  # Ensure computation completes
optax_adam_runtime = time() - start_time

optax_adam_mse = loss_fn(θ, x_validate, y_validate)
print(f"Trained model with JAX and Optax ADAM in {optax_adam_runtime:.2f} seconds.")
print(f"Final MSE on validation data = {optax_adam_mse:.6f}")
```

Here's a visualization of the result.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(),
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
    'Validation MSE': [
        keras_mse,
        jax_mse,
        optax_sgd_mse,
        optax_adam_mse
    ]
}

df = pd.DataFrame(results)
df.style.format({'Runtime (s)': '{:.2f}', 'Validation MSE': '{:.2f}'})
```


All methods achieve similar validation MSE values (around 0.043-0.045).

At the time of writing, the MSEs from plain vanilla Optax and our own hand-coded SGD routine are identical.

The ADAM optimizer achieves slightly better MSE by using adaptive learning rates.

Still, our hand-coded algorithm does very well compared to this high-quality optimizer.

Note also that the pure JAX implementations are significantly faster than Keras.

This is because JAX can JIT-compile the entire training loop.

Not surprisingly, Keras has more overhead from its abstraction layers.
