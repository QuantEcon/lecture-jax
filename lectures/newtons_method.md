---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Newton’s Method via JAX

```{include} _admonition/gpu.md
```

## Overview

One of the key features of JAX is automatic differentiation.

While other software packages also offer this feature, the JAX version is
particularly powerful because it integrates so closely with other core
components of JAX, such as accelerated linear algebra, JIT compilation and
parallelization.

The application of automatic differentiation we consider is computing economic equilibria via Newton's method.

Newton's method is a relatively simple root and fixed point solution algorithm, which we discussed 
in [a more elementary QuantEcon lecture](https://python.quantecon.org/newton_method.html).

JAX is almost ideally suited to implementing Newton's method efficiently, even
in high dimensions.

We use the following imports in this lecture

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from scipy.optimize import root
import matplotlib.pyplot as plt
```

Let's check the GPU we are running

```{code-cell} ipython3
!nvidia-smi
```

## Newton in one dimension

As a warm up, let's implement Newton's method in JAX for a simple
one-dimensional root-finding problem.

Let $f$ be a function from $\mathbb R$ to itself.

A **root** of $f$ is an $x \in \mathbb R$ such that $f(x)=0$.

[Recall](https://python.quantecon.org/newton_method.html) that Newton's method for solving for the root of $f$ involves iterating with the map $q$ defined by

$$ 
    q(x) = x - \frac{f(x)}{f'(x)} 
$$


Here is a function called `newton` that takes a function $f$ plus a scalar value $x_0$,
iterates with $q$ starting from $x_0$, and returns an approximate fixed point.

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

The code above uses automatic differentiation to calculate $f'$ via the call to `jax.grad`.

Let's test our `newton` routine on the function shown below.

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

Here we go

```{code-cell} ipython3
newton(f, 0.2)
```

This number looks to be close to the root, given the figure.



## An Equilibrium Problem

Now let's move up to higher dimensions.

First we describe a market equilibrium problem we will solve with JAX via root-finding.

The market is for $n$ goods.

(We are extending a two-good version of the market from [an earlier lecture](https://python.quantecon.org/newton_method.html).)

The supply function for the $i$-th good is

$$
    q^s_i (p) = b_i \sqrt{p_i}
$$

which we write in vector form as

$$
    q^s (p) =b \sqrt{p}
$$

(Here $\sqrt{p}$ is the square root of each $p_i$ and $b \sqrt{p}$ is the vector
formed by taking the pointwise product $b_i \sqrt{p_i}$ at each $i$.)

The demand function is

$$
    q^d (p) = \exp(- A p) + c
$$

(Here $A$ is an $n \times n$ matrix containing parameters, $c$ is an $n \times
1$ vector and the $\exp$ function acts pointwise (element-by-element) on the
vector $- A p$.)

The excess demand function is

$$
    e(p) = \exp(- A p) + c - b \sqrt{p}
$$

An **equilibrium price** vector is an $n$-vector $p$ such that $e(p) = 0$.

The function below calculates the excess demand for given parameters

```{code-cell} ipython3
def e(p, A, b, c):
    return jnp.exp(- A @ p) + c - b * jnp.sqrt(p)
```

## Computation

In this section we describe and then implement the solution method.


### Newton's Method

We use a multivariate version of Newton's method to compute the equilibrium price.

The rule for updating a guess $p_n$ of the equilibrium price vector is

```{math}
:label: multi-newton
p_{n+1} = p_n - J_e(p_n)^{-1} e(p_n)
```

Here $J_e(p_n)$ is the Jacobian of $e$ evaluated at $p_n$.

Iteration starts from initial guess $p_0$.

Instead of coding the Jacobian by hand, we use automatic differentiation via `jax.jacobian()`.

```{code-cell} ipython3
def newton(f, x_0, tol=1e-5, max_iter=15):
    """
    A multivariate Newton root-finding routine.

    """
    x = x_0
    f_jac = jax.jacobian(f)
    @jax.jit
    def q(x):
        " Updates the current guess. "
        return x - jnp.linalg.solve(f_jac(x), f(x))
    error = tol + 1
    n = 0
    while error > tol:
        n += 1
        if(n > max_iter):
            raise Exception('Max iteration reached without convergence')
        y = q(x)
        error = jnp.linalg.norm(x - y)
        x = y
        print(f'iteration {n}, error = {error}')
    return x
```

### Application

Let's now apply the method just described to investigate a large market with 5,000 goods.

We randomly generate the matrix $A$ and set the parameter vectors $b, c$ to $1$.

```{code-cell} ipython3
dim = 5_000
seed = 32

# Create a random matrix A and normalize the rows to sum to one
key = jax.random.PRNGKey(seed)
A = jax.random.uniform(key, [dim, dim])
s = jnp.sum(A, axis=0)
A = A / s

# Set up b and c
b = jnp.ones(dim)
c = jnp.ones(dim)
```

Here's our initial condition $p_0$

```{code-cell} ipython3
init_p = jnp.ones(dim)
```

By combining the power of Newton's method, JAX accelerated linear algebra,
automatic differentiation, and a GPU, we obtain a relatively small error for
this high-dimensional problem in just a few seconds:

```{code-cell} ipython3
%%time
p = newton(lambda p: e(p, A, b, c), init_p).block_until_ready()
```

We run it again to eliminate the compilation time.

```{code-cell} ipython3
%%time
p = newton(lambda p: e(p, A, b, c), init_p).block_until_ready()
```

Here's the size of the error:

```{code-cell} ipython3
jnp.max(jnp.abs(e(p, A, b, c)))
```

With the same tolerance, SciPy's `root` function takes much longer to run,
even with the Jacobian supplied.

```{code-cell} ipython3
%%time
solution = root(lambda p: e(p, A, b, c),
                init_p,
                jac=lambda p: jax.jacobian(e)(p, A, b, c),
                method='hybr',
                tol=1e-5)
```

```{code-cell} ipython3
%%time
solution = root(lambda p: e(p, A, b, c),
                init_p,
                jac=lambda p: jax.jacobian(e)(p, A, b, c),
                method='hybr',
                tol=1e-5)
```

The result is also slightly less accurate:

```{code-cell} ipython3
p = solution.x
jnp.max(jnp.abs(e(p, A, b, c)))
```

## Exercises

```{exercise-start}
:label: newton_ex1
```

Consider a three-dimensional extension of [the Solow fixed point
problem](https://python.quantecon.org/newton_method.html#the-solow-model) with

$$
A = \begin{pmatrix}
            2 & 3 & 3 \\
            2 & 4 & 2 \\
            1 & 5 & 1 \\
        \end{pmatrix},
            \quad
s = 0.2, \quad α = 0.5, \quad δ = 0.8
$$

As before the law of motion is

```{math}
    k_{t+1} = g(k_t) \quad \text{where} \quad
    g(k) := sAk^\alpha + (1-\delta) k
```

However $k_t$ is now a $3 \times 1$ vector.

Solve for the fixed point using Newton's method with the following initial values:

$$
\begin{aligned}
    k1_{0} &= (1, 1, 1) \\
    k2_{0} &= (3, 5, 5) \\
    k3_{0} &= (50, 50, 50)
\end{aligned}
$$

````{hint}
:class: dropdown
- The computation of the fixed point is equivalent to computing $k^*$ such that $f(k^*) - k^* = 0$.
- If you are unsure about your solution, you can start with the solved example:
```{math}
A = \begin{pmatrix}
            2 & 0 & 0 \\
            0 & 2 & 0 \\
            0 & 0 & 2 \\
        \end{pmatrix}
```
with $s = 0.3$, $α = 0.3$, and $δ = 0.4$ and starting value:
```{math}
k_0 = (1, 1, 1)
```
The result should converge to the [analytical solution](https://python.quantecon.org/newton_method.html#solved-k).
````

```{exercise-end}
```


```{solution-start} newton_ex1
:class: dropdown
```

Let's first define the parameters for this problem

```{code-cell} ipython3
A = jnp.array([[2.0, 3.0, 3.0],
               [2.0, 4.0, 2.0],
               [1.0, 5.0, 1.0]])
s = 0.2
α = 0.5
δ = 0.8
initLs = [jnp.ones(3),
          jnp.array([3.0, 5.0, 5.0]),
          jnp.repeat(50.0, 3)]
```

Then we define the multivariate version of the formula for the [law of motion of capital](https://python.quantecon.org/newton_method.html#solow)

```{code-cell} ipython3
def multivariate_solow(k, A=A, s=s, α=α, δ=δ):
    return s * jnp.dot(A, k**α) + (1 - δ) * k
```

Let's run through each starting value and see the output

```{code-cell} ipython3
attempt = 1
for init in initLs:
    print(f'Attempt {attempt}: Starting value is {init} \n')
    %time k = newton(lambda k: multivariate_solow(k) - k, \
                     init).block_until_ready()
    print('-'*64)
    attempt += 1
```

We find that the results are invariant to the starting values.

But the number of iterations it takes to converge is dependent on the starting values.

Let substitute the output back into the formulate to check our last result

```{code-cell} ipython3
multivariate_solow(k) - k
```

Note the error is very small.

We can also test our results on the known solution

```{code-cell} ipython3
A = jnp.array([[2.0, 0.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 2.0]])
s = 0.3
α = 0.3
δ = 0.4
init = jnp.repeat(1.0, 3)
%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k, \
                 init).block_until_ready()
```

```{code-cell} ipython3
%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k, \
                 init).block_until_ready()
```

The result is very close to the true solution but still slightly different.

We can increase the precision of the floating point numbers and restrict the tolerance to obtain a more accurate approximation (see detailed discussion in the [lecture on JAX](https://python-programming.quantecon.org/jax_intro.html#differences))

```{code-cell} ipython3
# We will use 64 bit floats with JAX in order to increase the precision.
jax.config.update("jax_enable_x64", True)
init = init.astype('float64')

%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k,\
                 init, tol=1e-7).block_until_ready()
```

```{code-cell} ipython3
%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k,\
                 init, tol=1e-7).block_until_ready()
```

We can see it steps towards a more accurate solution.

```{solution-end}
```


```{exercise-start}
:label: newton_ex2
```

In this exercise, let's try different initial values and check how Newton's method responds to different starting points.

Let's define a three-good problem with the following default values:

$$
A = \begin{pmatrix}
            0.2 & 0.1 & 0.7 \\
            0.3 & 0.2 & 0.5 \\
            0.1 & 0.8 & 0.1 \\
        \end{pmatrix},
            \qquad
b = \begin{pmatrix}
            1 \\
            1 \\
            1
        \end{pmatrix}
    \qquad \text{and} \qquad
c = \begin{pmatrix}
            1 \\
            1 \\
            1
        \end{pmatrix}
$$

For this exercise, use the following extreme price vectors as initial values:

$$

\begin{aligned}
    p1_{0} &= (5, 5, 5) \\
    p2_{0} &= (1, 1, 1) \\
    p3_{0} &= (4.5, 0.1, 4)
\end{aligned}
$$

Set the tolerance to $10^{-15}$ for more accurate output.


```{hint}
:class: dropdown
Similar to [exercise 1](newton_ex1), enabling `float64` for JAX can improve the precision of our results.
```


```{exercise-end}
```

```{solution-start} newton_ex2
:class: dropdown
```

Define parameters and initial values

```{code-cell} ipython3
A = jnp.array([
    [0.2, 0.1, 0.7],
    [0.3, 0.2, 0.5],
    [0.1, 0.8, 0.1]
])
b = jnp.array([1.0, 1.0, 1.0])
c = jnp.array([1.0, 1.0, 1.0])
initLs = [jnp.repeat(5.0, 3),
          jnp.array([4.5, 0.1, 4.0])]
```

Let’s run through each initial guess and check the output

```{code-cell} ipython3
attempt = 1
for init in initLs:
    print(f'Attempt {attempt}: Starting value is {init} \n')
    init = init.astype('float64')
    %time p = newton(lambda p: e(p, A, b, c), \
                 init, \
                 tol=1e-15, max_iter=15).block_until_ready()
    print('-'*64)
    attempt +=1
```

We can find that Newton's method may fail for some starting values.

Sometimes it may take a few initial guesses to achieve convergence.

Substitute the result back to the formula to check our result

```{code-cell} ipython3
e(p, A, b, c)
```

We can see the result is very accurate.

```{solution-end}
```
