---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# An Introduction to JAX


```{include} _admonition/gpu.md
```

This lecture provides a short introduction to [Google JAX](https://github.com/google/jax).


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
linalg.solve(B, A)
```

```{code-cell} ipython3
linalg.eigh(B)  # Computes eigenvalues and eigenvectors
```

### Differences


One difference between NumPy and JAX is that, when running on a GPU, JAX uses 32 bit floats by default.  

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
a
```

```{code-cell} ipython3
:tags: [raises-exception]

a[0] = 1
```

In line with immutability, JAX does not support inplace operations:

```{code-cell} ipython3
a = np.array((2, 1))
a.sort()
a
```

```{code-cell} ipython3
a = jnp.array((2, 1))
a_new = a.sort()
a, a_new
```

The designers of JAX chose to make arrays immutable because JAX uses a
functional programming style.  More on this below.  

Note that, while mutation is discouraged, it is in fact possible with `at`, as in

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
id(a)
```

```{code-cell} ipython3
a
```

```{code-cell} ipython3
a.at[0].set(1)
```

We can check that the array is mutated by verifying its identity is unchanged:

```{code-cell} ipython3
id(a)
```

## Random Numbers

Random numbers are also a bit different in JAX, relative to NumPy.  Typically, in JAX, the state of the random number generator needs to be controlled explicitly.

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
random.normal(key, (5, ))
```

## JIT Compilation


The JAX JIT compiler accelerates logic within functions by fusing linear
algebra operations into a single, highly optimized kernel that the host can
launch on the GPU / TPU (or CPU if no accelerator is detected).


Consider the following pure Python function.

```{code-cell} ipython3
def f(x, p=1000):
    return sum((k*x for k in range(p)))
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

```{note}
Here, in order to measure actual speed, we use the `block_until_ready()` method 
to hold the interpreter until the results of the computation are returned from
the device.

This is necessary because JAX uses asynchronous dispatch, which allows the
Python interpreter to run ahead of GPU computations.

```

This code is not particularly fast.  

While it is run on the GPU (since `x` is a JAX array), each vector `k * x` has to be instantiated before the final sum is computed.

If we JIT-compile the function with JAX, then the operations are fused and no intermediate arrays are created.

```{code-cell} ipython3
f_jit = jax.jit(f)   # target for JIT compilation
```

Let's run once to compile it:

```{code-cell} ipython3
f_jit(x)
```

And now let's time it.

```{code-cell} ipython3
%time f_jit(x).block_until_ready()
```

Note the large speed gain.


## Functional Programming

From JAX's documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has “una anima di pura programmazione funzionale”.*


In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure.
    
A pure function will always return the same result if invoked with the same inputs.

In particular, a pure function has

* no dependence on global variables and
* no side effects


JAX will not usually throw errors when compiling impure functions but execution becomes unpredictable.

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

Changing the dimension of the input triggers a fresh compilation of the function, at which time the change in the value of `a` takes effect:

```{code-cell} ipython3
x = np.ones(3)
```

```{code-cell} ipython3
f(x)
```

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

## Writing vectorized code

Writing fast JAX code requires shifting repetitive tasks from loops to array processing operations, so that the JAX compiler can easily understand the whole operation and generate more efficient machine code.

This procedure is called **vectorization** or **array programming**, and will be familiar to anyone who has used NumPy or MATLAB.

In most ways, vectorization is the same in JAX as it is in NumPy.

But there are also some differences, which we highlight here.

As a running example, consider the function

$$
    f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
$$

Suppose that we want to evaluate this function on a square grid of $x$ and $y$ points and then plot it.

To clarify, here is the slow `for` loop version.

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

+++

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

To get the right shape and the correct nested for loop calculation, we can use a `meshgrid` operation designed for this purpose:

```{code-cell} ipython3
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

Now we get what we want and the execution time is very fast.

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh) 
```

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh) 
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
z_mesh = f(x_mesh, y_mesh) 
```

```{code-cell} ipython3
%%time
z_mesh = f(x_mesh, y_mesh) 
```

But there is one problem here: the mesh grids use a lot of memory.

```{code-cell} ipython3
x_mesh.nbytes + y_mesh.nbytes
```

By comparison, the flat array `x` is just

```{code-cell} ipython3
x.nbytes  # and y is just a pointer to x
```

This extra memory usage can be a big problem in actual research calculations.

So let's try a different approach using [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) 

+++

First we vectorize `f` in `y`.

```{code-cell} ipython3
f_vec_y = jax.vmap(f, in_axes=(None, 0))  
```

In the line above, `(None, 0)` indicates that we are vectorizing in the second argument, which is `y`.

Next, we vectorize in the first argument, which is `x`.

```{code-cell} ipython3
f_vec = jax.vmap(f_vec_y, in_axes=(0, None))
```

With this construction, we can now call the function $f$ on flat (low memory) arrays.

```{code-cell} ipython3
%%time
z_vmap = f_vec(x, y)
```

```{code-cell} ipython3
%%time
z_vmap = f_vec(x, y)
```

The execution time is essentially the same as the mesh operation but we are using much less memory.

And we produce the correct answer:

```{code-cell} ipython3
jnp.allclose(z_vmap, z_mesh)
```

## Exercises



```{exercise-start}
:label: jax_intro_ex1
```

Recall that Newton's method for solving for the root of $f$ involves iterating on 


$$ 
    q(x) = x - \frac{f(x)}{f'(x)} 
$$

Write a function called `newton` that takes a function $f$ plus a guess $x_0$ and returns an approximate fixed point.

Your `newton` implementation should use automatic differentiation to calculate $f'$.

Test your `newton` method on the function shown below.

```{code-cell} ipython3
f = lambda x: jnp.sin(4 * (x - 1/4)) + x + x**20 - 1
x = jnp.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f(x)$')
ax.axhline(ls='--', c='k')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)
ax.legend(fontsize=12)
plt.show()
```

```{exercise-end}
```

```{solution-start} jax_intro_ex1
:class: dropdown
```

Here's a suitable function:

```{code-cell} ipython3
def newton(f, x_0, tol=1e-5):
    f_prime = jax.grad(f)
    def q(x):
        return x - f(x) / f_prime(x)

    error = tol + 1
    x = x_0
    while error > tol:
        y = q(x)
        error = abs(x - y)
        x = y
        
    return x
```

Let's try it:

```{code-cell} ipython3
newton(f, 0.2)
```

This number looks good, given the figure.


```{solution-end}
```



```{exercise-start}
:label: jax_intro_ex2
```

In {ref}`an earlier exercise on parallelization <jax_intro_ex1>`, we used Monte
Carlo to price a European call option.

The code was accelerated by Numba-based multithreading.

Try writing a version of this operation for JAX, using all the same
parameters.

If you are running your code on a GPU, you should be able to achieve
significantly faster exection.


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
compute_call_price_jax()
```

And now let's time it:

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

```{solution-end}
```
