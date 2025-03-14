---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Neural Network Regression with JAX and Optax

```{include} _admonition/gpu.md
```

In a [previous lecture](keras), we showed how to implement regression using a neural network via the popular deep learning library [Keras](https://keras.io/).

In this lecture, we solve the same problem using pure JAX instead.

The objective is to understand the nuts and bolts of the exercise better, as
well as to explore more features of JAX.

The lecture proceeds in three stages:

1. We repeat the Keras exercise, to give ourselves a benchmark.
2. We solve the same problem in pure JAX, using pytree operations and gradient descent.
3. We solve the same problem using a combination of JAX and [Optax](https://optax.readthedocs.io/en/latest/index.html), an optimization library build for JAX.

We begin with imports and installs.

```{code-cell} ipython3
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
```

```{code-cell} ipython3
:tags: [hide-output]

!pip install keras
```

```{code-cell} ipython3
:tags: [hide-output]

!pip install optax
```

```{code-cell} ipython3
os.environ['KERAS_BACKEND'] = 'jax'
```

```{code-cell} ipython3
import keras
from keras import Sequential
from keras.layers import Dense
import optax
```

## Set Up

Let's hardcode some of the learning-related constants we'll use across all
implementations.

```{code-cell} ipython3
EPOCHS = 4000           # Number of passes through the data set
DATA_SIZE = 400         # Sample size
NUM_LAYERS = 4          # Depth of the network
OUTPUT_DIM = 10         # Output dimension of input and hidden layers
LEARNING_RATE = 0.001   # Learning rate for gradient descent
```

The next piece of code is repeated from [our Keras lecture](keras) and generates
the data.

```{code-cell} ipython3
def generate_data(x_min=0,           
                  x_max=5,          
                  data_size=DATA_SIZE,
                  seed=1234): # Default size for dataset
    np.random.seed(seed)
    x = np.linspace(x_min, x_max, num=data_size)
    ϵ = 0.2 * np.random.randn(data_size)
    y = x**0.5 + np.sin(x) + ϵ
    # Return observations as column vectors 
    x, y = [np.reshape(z, (data_size, 1)) for z in (x, y)]
    return x, y
```

## Training with Keras 


We repeat the Keras training exercise from [our Keras lecture](keras) as a
benchmark.

The code is essentially the same, although written slightly more succinctly.

Here is a function to build the model.

```{code-cell} ipython3
def build_keras_model(num_layers=NUM_LAYERS, 
                      activation_function='tanh'):
    model = Sequential()
    # Add layers to the network sequentially, from inputs towards outputs
    for i in range(NUM_LAYERS-1):
        model.add(
           Dense(units=OUTPUT_DIM, 
                 activation=activation_function)
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

Here is a function to train the model.

```{code-cell} ipython3
def train_keras_model(model, x, y, x_validate, y_validate):
    print(f"Training NN using Keras.")
    training_history = model.fit(
        x, y, 
        batch_size=max(x.shape), 
        verbose=0,
        epochs=EPOCHS, 
        validation_data=(x_validate, y_validate)
    )
    mse = model.evaluate(x_validate, y_validate, verbose=2)
    print(f"Trained Keras model with final MSE on validation data = {mse}")
    return model, training_history
```

The next function visualizes the prediction.

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

Here's a function to run all the routines above.

```{code-cell} ipython3
def keras_run_all():
    model = build_keras_model()
    x, y = generate_data()
    x_validate, y_validate = generate_data()
    model, training_history = train_keras_model(
            model, x, y, x_validate, y_validate
    )
    plot_keras_output(model, x, y, x_validate, y_validate)
```

Let's put it to work:

```{code-cell} ipython3
%time keras_run_all()
```

We've seen this figure before and we note the relatively low final MSE.


## Training with JAX 

For the JAX implementation, we need to construct the network ourselves, as a map
from inputs to outputs.

We'll use the same network structure we used for the Keras implementation.

### Background and set up


The neural network as the form 

$$
    f(\theta, x) 
    = (A_3 \circ \sigma \circ A_2 \circ \sigma \circ A_1 \circ \sigma A_0)(x)
$$

Here 

* $\circ$ means composition of maps,
* $\sigma$ is the activation funcion -- in our case, $\tanh$, and
* $A_i$ represents the affine map $A_i x = W_i x + b_i$.

Each matrix $W_i$ is called a **weight matrix** and each vector $b_i$ is called **bias** term.

The symbol $\theta$ represents the entire collection of parameters:

$$
    \theta = (W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3)
$$

In fact, when we implement the affine map $A_i x = W_i x + b_i$, we will work
with row vectors rather than column vectors, so that

* $x$ and $b_i$ are stored as row vectors, and
* the mapping is executed as $x @ W + b$.

This is because Python numerical operations are row-major rather than column-major, so that row-based operations tend to be more efficient.

Here's a function to initialize parameters.

The parameter ``vector'' `θ`  will be stored as a list of dicts.

```{code-cell} ipython3
def initialize_params(seed=1234):
    """
    Generate an initial parameterization for a feed forward neural network with
    number of layers = NUM_LAYERS.  Each of the hidden layers have OUTPUT_DIM
    units.
    """
    k = OUTPUT_DIM
    shapes = (
        (1, k),  # W_0.shape
        (k, k),  # W_1.shape
        (k, k),  # W_2.shape
        (k, 1)   # W_3.shape
    )   
    np.random.seed(seed)
    # A function to generate weight matrices
    def w_init(m, n):
        return np.random.normal(size=(m, n)) * np.sqrt(2 / m)
    # Build list of dicts, each containing a (weight, bias) pair
    θ = []
    for w_shape in shapes:
        m, n = w_shape
        θ.append(dict(W=w_init(m, n), b=np.ones((1, n))) )
    return θ
```

### Coding the network

Here's our implementation of $f$:

```{code-cell} ipython3
@jax.jit
def f(θ, x):
    """
    Perform a forward pass over the network to evaluate f(θ, x).
    The state x is stored and iterated on as a row vector.
    """
    *hidden, last = θ
    for layer in hidden:
        W, b = layer['W'], layer['b']
        x = jnp.tanh(x @ W + b)
    W, b = last['W'], last['b']
    x = x @ W + b
    return x 
```

The function $f$ is appropriately vectorized, so that we can pass in the entire
set of input observations as `x` and return the predicted vector of outputs `y_hat = f(θ, x)`
corresponding  to each data point.

The loss function is mean squared error, the same as the Keras case.

```{code-cell} ipython3
@jax.jit
def loss_fn(θ, x, y):
    "Loss is mean squared error."
    return jnp.mean((f(θ, x) - y)**2)
```

We'll use its gradient to do stochastic gradient descent.

(Technically, we will be doing gradient descent, rather than stochastic
gradient descent, since will not randomize over sample points when we
evaluate the gradient.)

The gradient below is with respect to the first argument `θ`.

```{code-cell} ipython3
loss_gradient = jax.jit(jax.grad(loss_fn))
```

The line above seems kind of magical, since we are differentiating with respect
to a parameter ``vector'' stored as a list of dictionaries containing arrays.

How can we differentiate with respect to such a complex object?

The answer is that the list of dictionaries is treated internally as a
[pytree](https://docs.jax.dev/en/latest/pytrees.html).

The JAX function `grad` understands how to 

1. extract the individual arrays (the ``leaves'' of the tree), 
2. compute derivatives with respect to each one, and 
3. pack the resulting derivatives into a pytree with the same structure as the parameter vector.


### Gradient descent

Using the above code, we can now write our rule for updating the parameters via gradient descent, which is the
algorithm we covered in our [lecture on autodiff](autodiff).

In this case, however, to keep things as simple as possible, we'll use a fixed learning rate for every iteration.

```{code-cell} ipython3
@jax.jit
def update_parameters(θ, x, y):
    λ = LEARNING_RATE 
    gradient = loss_gradient(θ, x, y)
    θ = jax.tree.map(lambda p, g: p - λ * g, θ, gradient)
    return θ
```

We are implementing the gradient descent update 

```
    new_params = current_params - learning_rate * gradient_of_loss_function
```

This is nontrivial for a complex structure such as a neural network, so how is
it done?

The key line in the function above is `Θ = jax.tree.map(lambda p, g: p - λ * g, θ, gradient)`.

The `jax.tree.map` function understands `θ` and `gradient` as pytrees of the
same structure and executes `p - λ * g` on the corresponding leaves of the pair
of trees.

This means that each weight matrix and bias vector is updated by gradient
descent, exactly as required.


Here's code that puts this all together.

```{code-cell} ipython3
def train_jax_model(θ, x, y, x_validate, y_validate):
    """
    Train model using gradient descent via JAX autodiff.
    """
    training_loss = np.empty(EPOCHS)
    validation_loss = np.empty(EPOCHS)
    for i in range(EPOCHS):
        training_loss[i] = loss_fn(θ, x, y)
        validation_loss[i] = loss_fn(θ, x_validate, y_validate)
        θ = update_parameters(θ, x, y)
    return θ, training_loss, validation_loss
```

### Execution

Let's run our code and see how it goes.

```{code-cell} ipython3
θ = initialize_params()
x, y = generate_data()
x_validate, y_validate = generate_data()
```

```{code-cell} ipython3
%%time 

θ, training_loss, validation_loss = train_jax_model(
    θ, x, y, x_validate, y_validate
)
```

This figure shows MSE across iterations:

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(range(EPOCHS), validation_loss, label='validation loss')
ax.legend()
plt.show()
```

Let's check the final MSE on the validation data, at the estimated parameters.

```{code-cell} ipython3
print(f"""
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
```

This MSE is not as low as we got for Keras, but we did quite well given how
simple our implementation is.

Here's a visualization of the quality of our fit.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(), 
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

## Using Optax

Our hand-coded optimization routine above was quite effective, but in practice
we might wish to use an optimization library written for JAX.

One such library is [Optax](https://optax.readthedocs.io/en/latest/).

Here's a training routine using Optax's stochastic gradient descent solver.

```{code-cell} ipython3
def train_jax_optax(θ, x, y):
    solver = optax.sgd(learning_rate=LEARNING_RATE)
    opt_state = solver.init(θ)
    for _ in range(EPOCHS):
        grad = loss_gradient(θ, x, y)
        updates, opt_state = solver.update(grad, opt_state, θ)
        θ = optax.apply_updates(θ, updates)
    return θ
```

Let's try running it.

```{code-cell} ipython3
%% time 

# Reset parameter vector
θ = initialize_params()
# Train network
θ = train_jax_optax(θ, x, y)
```

The results are similar to our hand-coded routine.

```{code-cell} ipython3
print(f"""
Completed training JAX model using Optax with SGD.
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
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

We can also consider using a slightly more sophisticated gradient-based method,
such as [ADAM](https://arxiv.org/pdf/1412.6980).


You will notice that the method is very similar.

```{code-cell} ipython3
def train_jax_optax(θ, x, y):
    solver = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = solver.init(θ)
    for _ in range(EPOCHS):
        grad = loss_gradient(θ, x, y)
        updates, opt_state = solver.update(grad, opt_state, θ)
        θ = optax.apply_updates(θ, updates)
    return θ
```

```{code-cell} ipython3
%% time 

# Reset parameter vector
θ = initialize_params()
# Train network
θ = train_jax_optax(θ, x, y)
```

Here's the MSE.

```{code-cell} ipython3
print(f"""
Completed training JAX model using Optax with ADAM.
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
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
