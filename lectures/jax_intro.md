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

# An Introduction to JAX


This lecture provides a short introduction to [Google JAX](https://github.com/google/jax).

Let's see if we have an active GPU:

```{code-cell} ipython3
!nvidia-smi
```

## JAX as a NumPy Replacement


One way to use JAX is as a plug-in NumPy replacement. Let's look at the
similarities and differences.

### Similarities


The following import is standard, replacing `import numpy as np`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

Now we can use `jnp` in place of `np` for the usual array operations:

```{code-cell} ipython3
a = jnp.asarray((1.0, 3.2, -1.5))
```

```{code-cell} ipython3
print(a)
```

```{code-cell} ipython3
print(jnp.sum(a))
```

```{code-cell} ipython3
print(jnp.mean(a))
```

```{code-cell} ipython3
print(jnp.dot(a, a))
```

```{code-cell} ipython3
print(a @ a)  # Equivalent
```

However, the array object `a` is not a NumPy array:

```{code-cell} ipython3
a
```

```{code-cell} ipython3
type(a)
```

Even scalar-valued maps on arrays return JAX arrays.

```{code-cell} ipython3
jnp.sum(a)
```

JAX arrays are also called "device arrays," where term "device" refers to a
hardware accelerator (GPU or TPU).

(In the terminology of GPUs, the "host" is the machine that launches GPU operations, while the "device" is the GPU itself.)



Operations on higher dimensional arrays are also similar to NumPy:

```{code-cell} ipython3
A = jnp.ones((2, 2))
B = jnp.identity(2)
A @ B
```

```{code-cell} ipython3
from jax.numpy import linalg
```

```{code-cell} ipython3
linalg.inv(B)   # Inverse of identity is identity
```

```{code-cell} ipython3
result = linalg.eigh(B)  # Computes eigenvalues and eigenvectors
result.eigenvalues
```

```{code-cell} ipython3
result.eigenvectors
```

### Differences


One difference between NumPy and JAX is that JAX currently uses 32 bit floats by default.  

This is standard for GPU computing and can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.  In these cases 64 bit floats can be enforced via the command

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

Let's check this works:

```{code-cell} ipython3
jnp.ones(3)
```

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.  

For example, with NumPy we can write

```{code-cell} ipython3
import numpy as np
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```{code-cell} ipython3
a[0] = 1
a
```

In JAX this fails:

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
a[0] = 1
```

```{code-cell} ipython3
:tags: [raises-exception]

a
```

The designers of JAX chose to make arrays immutable because JAX uses a
functional programming style.  More on this below.  

However, JAX provides a functionally pure equivalent of in-place array modification
using the [`at` method](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html).

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
id(a)
```

```{code-cell} ipython3
a
```

Applying `at[0].set(1)` returns a new copy of `a` with the first element set to 1

```{code-cell} ipython3
a = a.at[0].set(1)
a
```

Inspecting the identifier of `a` shows that it has been reassigned

```{code-cell} ipython3
id(a)
```

## Random Numbers

Random numbers are also a bit different in JAX, relative to NumPy.  

Typically, in JAX, the state of the random number generator needs to be controlled explicitly.

(This is also related to JAX's functional programming paradigm, discussed below.  JAX does not typically work with objects that maintain state, such as the state of a random number generator.)

```{code-cell} ipython3
import jax.random as random
```

First we produce a key, which seeds the random number generator.

```{code-cell} ipython3
key = random.PRNGKey(1)
```

```{code-cell} ipython3
type(key)
```

```{code-cell} ipython3
print(key)
```

Now we can use the key to generate some random numbers:

```{code-cell} ipython3
x = random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```{code-cell} ipython3
random.normal(key, (3, 3))
```

To produce a (quasi-) independent draw, best practice is to "split" the existing key:

```{code-cell} ipython3
key, subkey = random.split(key)
```

```{code-cell} ipython3
random.normal(key, (3, 3))
```

```{code-cell} ipython3
random.normal(subkey, (3, 3))
```

The function below produces `k` (quasi-) independent random `n x n` matrices using this procedure.

```{code-cell} ipython3
def gen_random_matrices(key, n, k):
    matrices = []
    for _ in range(k):
        key, subkey = random.split(key)
        matrices.append(random.uniform(subkey, (n, n)))
    return matrices
```

```{code-cell} ipython3
matrices = gen_random_matrices(key, 2, 2)
for A in matrices:
    print(A)
```

One point to remember is that JAX expects tuples to describe array shapes, even for flat arrays.  Hence, to get a one-dimensional array of normal random draws we use `(len, )` for the shape, as in

```{code-cell} ipython3
random.normal(key, (5,))   # not random.normal(key, 5)
```

## JIT compilation

The JAX just-in-time (JIT) compiler accelerates logic within functions by fusing linear
algebra operations into a single optimized kernel that the host can
launch on the GPU / TPU (or CPU if no accelerator is detected.)

### A first example

To see the JIT compiler in action, consider the following function.

```{code-cell} ipython3
def f(x):
    a = 3*x + jnp.sin(x) + jnp.cos(x**2) - jnp.cos(2*x) - x**2 * 0.4 * x**1.5
    return jnp.sum(a)
```

Let's build an array to call the function on.

```{code-cell} ipython3
n = 50_000_000
x = jnp.ones(n)
```

How long does the function take to execute?

```{code-cell} ipython3
%time f(x).block_until_ready()
```

(In order to measure actual speed, we use `block_until_ready()` method 
to hold the interpreter until the results of the computation are returned from
the device. This is necessary because JAX uses asynchronous dispatch, which
allows the Python interpreter to run ahead of GPU computations.)



The code doesn't run as fast as we might hope, given that it's running on a GPU.

But if we run it a second time it becomes much faster:

```{code-cell} ipython3
%time f(x).block_until_ready()
```

This is because the built in functions like `jnp.cos` are JIT compiled and the
first run includes compile time.



### When does JAX recompile?

You might remember that Numba recompiles if we change the types of variables in a function call.

JAX recompiles more often --- in particular, it recompiles every time we change array sizes.

For example, let's try

```{code-cell} ipython3
m = n + 1
y = jnp.ones(m)
```

```{code-cell} ipython3
%time f(y).block_until_ready()
```

Notice that the execution time increases, because now new versions of 
the built-ins like `jnp.cos` are being compiled, specialized to the new array
size.

If we run again, the code is dispatched to the correct compiled version and we
get faster execution.

```{code-cell} ipython3
%time f(y).block_until_ready()
```

Why does JAX generate fresh machine code every time we change the array size???



The compiled versions for the previous array size are still available in memory
too, and the following call is dispatched to the correct compiled code.

```{code-cell} ipython3
%time f(x).block_until_ready()
```

### Compiling user-built functions

We can instruct JAX to compile entire functions that we build.

For example, consider

```{code-cell} ipython3
def g(x):
    y = jnp.zeros_like(x)
    for i in range(10):
        y += x**i
    return y
```

```{code-cell} ipython3
n = 1_000_000
x = jnp.ones(n)
```

Let's time it.

```{code-cell} ipython3
%time g(x)
```

```{code-cell} ipython3
%time g(x)
```

```{code-cell} ipython3
g_jit = jax.jit(g)   # target for JIT compilation
```

Let's run once to compile it:

```{code-cell} ipython3
g_jit(x)
```

And now let's time it.

```{code-cell} ipython3
%time g_jit(x).block_until_ready()
```

Note the speed gain.

This is because 

1. the loop is compiled and
2. the array operations are fused and no intermediate arrays are created.


Incidentally, a more common syntax when targetting a function for the JIT compiler is

```{code-cell} ipython3
@jax.jit
def g_jit_2(x):
    y = jnp.zeros_like(x)
    for i in range(10):
        y += x**i
    return y
```

```{code-cell} ipython3
%time g_jit_2(x).block_until_ready()
```

```{code-cell} ipython3
%time g_jit_2(x).block_until_ready()
```

## Functional Programming

From JAX's documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has “una anima di pura programmazione funzionale”.*


In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure.
    
A pure function will always return the same result if invoked with the same inputs.

In particular, a pure function has

* no dependence on global variables and
* no side effects



### Example: Python/NumPy/Numba style code is not pure



Here's an example to show that NumPy functions are not pure:

```{code-cell} ipython3
np.random.randn()
```

```{code-cell} ipython3
np.random.randn()
```

This fails the test: a function returns the same result when called on the same inputs.

The issue is that the function maintains internal state between function calls --- the state of the random number generator.



Here's a function that fails to be pure because it modifies external state.

```{code-cell} ipython3
def double_input(x):   # Not pure -- side effects
    x[:] = 2 * x
    return None

x = np.ones(5)
x
```

```{code-cell} ipython3
double_input(x)
x
```

Here's a pure version:

```{code-cell} ipython3
def double_input(x):
    y = 2 * x
    return y
```

The following function is also not pure, since it modifies a global variable (similar to the last example).

```{code-cell} ipython3
a = 1
def f():
    global a
    a += 1
    return None
```

```{code-cell} ipython3
a
```

```{code-cell} ipython3
f()
```

```{code-cell} ipython3
a
```

### Compiling impure functions

JAX does not insist on pure functions.

For example, JAX will not usually throw errors when compiling impure functions 

However, execution becomes unpredictable!

Here's an illustration of this fact, using global variables:

```{code-cell} ipython3
a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```{code-cell} ipython3
x = jnp.ones(2)
```

```{code-cell} ipython3
x
```

```{code-cell} ipython3
f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected --- as long as the same compiled version is called.

```{code-cell} ipython3
a = 42
```

```{code-cell} ipython3
f(x)
```

Notice that the change in the value of `a` takes effect in the code below:

```{code-cell} ipython3
x = jnp.ones(3)
```

```{code-cell} ipython3
f(x)
```

Can you explain why?

Moral of the story: write pure functions when using JAX!

## Gradients

JAX can use automatic differentiation to compute gradients.

This can be extremely useful for optimization and solving nonlinear systems.

We will see significant applications later in this lecture series.

For now, here's a very simple illustration involving the function

```{code-cell} ipython3
def f(x):
    return (x**2) / 2
```

Let's take the derivative:

```{code-cell} ipython3
f_prime = jax.grad(f)
```

```{code-cell} ipython3
f_prime(10.0)
```

Let's plot the function and derivative, noting that $f'(x) = x$.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_grid = jnp.linspace(-4, 4, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, [f_prime(x) for x in x_grid], label="$f'$")
ax.legend(loc='upper center')
plt.show()
```

We defer further exploration of automatic differentiation with JAX until {doc}`autodiff`.

## Writing vectorized code

Writing fast JAX code requires shifting repetitive tasks from loops to array processing operations, so that the JAX compiler can easily understand the whole operation and generate more efficient machine code.

This procedure is called **vectorization** or **array programming**, and will be familiar to anyone who has used NumPy or MATLAB.

In some ways, vectorization is the same in JAX as it is in NumPy.

But there are also major differences, which we highlight here.

As a running example, consider the function

$$
    f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
$$

Suppose that we want to evaluate this function on a square grid of $x$ and $y$ points.


### A slow version with loops

To clarify, here is the slow `for` loop version, which we run in a setting where `len(x) = len(y)` is very small.

```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)

n = 80
x = jnp.linspace(-2, 2, n)
y = x

z_loops = np.empty((n, n))
```

```{code-cell} ipython3
%%time
for i in range(n):
    for j in range(n):
        z_loops[i, j] = f(x[i], y[j])
```

Even for this very small grid, the run time is extremely slow.

(Notice that we used a NumPy array for `z_loops` because we wanted to write to it.)



OK, so how can we do the same operation in vectorized form?

If you are new to vectorization, you might guess that we can simply write

```{code-cell} ipython3
z_bad = f(x, y)
```

But this gives us the wrong result because JAX doesn't understand the nested for loop.

```{code-cell} ipython3
z_bad.shape
```

Here is what we actually wanted:

```{code-cell} ipython3
z_loops.shape
```

### Vectorization attempt 1


To get the right shape and the correct nested for loop calculation, we can use a `meshgrid` operation that originated in MATLAB and was replicated in NumPy and then JAX:

```{code-cell} ipython3
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

Now we get what we want and the execution time is very fast.

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's confirm that we got the right answer.

```{code-cell} ipython3
jnp.allclose(z_mesh, z_loops)
```

Now we can set up a serious grid and run the same calculation (on the larger grid) in a short amount of time.

```{code-cell} ipython3
n = 6000
x = jnp.linspace(-2, 2, n)
y = x
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's run again to get rid of compile time.

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

But there is one problem here: the mesh grids use a lot of memory.

```{code-cell} ipython3
(x_mesh.nbytes + y_mesh.nbytes) / 1_000_000  # MB of memory
```

By comparison, the flat array `x` is just

```{code-cell} ipython3
x.nbytes / 1_000_000   # and y is just a pointer to x
```

This extra memory usage can be a big problem in actual research calculations.

### Vectorization attempt 2


We can achieve a similar effect through NumPy style broadcasting rules.

```{code-cell} ipython3
x_reshaped = jnp.reshape(x, (n, 1))   # Give x another dimension (column)
y_reshaped = jnp.reshape(y, (1, n))   # Give y another dimension (row)
```

When we evaluate $f$ on these reshaped arrays, we replicate the nested for loops in the original version.

```{code-cell} ipython3
%time z_reshaped = f(x_reshaped, y_reshaped)
```

Let's check that we got the same result

```{code-cell} ipython3
jnp.allclose(z_reshaped, z_mesh)
```

The memory usage for the inputs is much more moderate.

```{code-cell} ipython3
(x_reshaped.nbytes + y_reshaped.nbytes) / 1_000_000
```

### Vectorization attempt 3


There's another approach to vectorization we can pursue, using [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)



It runs out that, when we are working with complex functions and operations, this `vmap` approach can be the easiest to implement.

It's also very memory parsimonious.



The first step is to vectorize the function `f` in `y`.

```{code-cell} ipython3
f_vec_y = jax.vmap(f, in_axes=(None, 0))  
```

In the line above, `(None, 0)` indicates that we are vectorizing in the second argument, which is `y`.

Next, we vectorize in the first argument, which is `x`.

```{code-cell} ipython3
f_vec = jax.vmap(f_vec_y, in_axes=(0, None))
```

Finally, we JIT-compile the result:

```{code-cell} ipython3
f_vec = jax.jit(f_vec)
```

With this construction, we can now call the function $f$ on flat (low memory) arrays.

```{code-cell} ipython3
%%time
z_vmap = f_vec(x, y).block_until_ready()
```

We run it again to eliminate compile time.

```{code-cell} ipython3
%%time
z_vmap = f_vec(x, y).block_until_ready()
```

Let's check we produce the correct answer:

```{code-cell} ipython3
jnp.allclose(z_vmap, z_mesh)
```

## Exercises


```{exercise-start}
:label: jax_intro_ex2
```

In the Exercise section of [a lecture on Numba and parallelization](https://python-programming.quantecon.org/parallelization.html), we used Monte Carlo to price a European call option.

The code was accelerated by Numba-based multithreading.

Try writing a version of this operation for JAX, using all the same
parameters.

If you are running your code on a GPU, you should be able to achieve
significantly faster execution.


```{exercise-end}
```


```{solution-start} jax_intro_ex2
:class: dropdown
```
Here is one solution:

```{code-cell} ipython3
M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           key=jax.random.PRNGKey(1)):

    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)
    for t in range(n):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))
        
    return β**n * expectation
```

Let's run it once to compile it:

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

And now let's time it:

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

```{solution-end}
```
