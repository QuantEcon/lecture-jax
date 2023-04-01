---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(newton_method)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```
```{index} single: python
```

## Inplementing Newton's Method with Google Jax

Continuing from the the [Newton's Method lecutre](https://python.quantecon.org/newton_method.html), we are going to solve the multidimensional problem with `Jax`. More information about Jax can be found [here](https://python-programming.quantecon.org/jax_intro.html).

We use the following imports in this lecture

```{code-cell} python3
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root
```

### The Two Goods Market Equilibrium

Let's have a quick recall for this problem, a more detailed explanation and derivation can be found at [A Two Goods Market Equilibrium](https://python.quantecon.org/newton_method.html#two_goods_market).

Assuming we have such market for two complementary goods where demand depends on the price of both components. 

We label them $\text{good}_0$ and $\text{ good}_1$, with price vector $p = (p_0, p_1)$.

Then the supply of good $i$ at price $p$ is,

$$ 
q^s_i (p) = b_i \sqrt{p_i} 
$$

and the demand of good $i$ at price $p$ is,

$$ 
q^d_i (p) = \text{exp}(-(a_{i0} p_0 + a_{i1} p_1)) + c_i
$$

Here $a_{ij}$, $b_i$ and $c_i$ are parameters for $n \times n$ square matrix $A$ and $n \times 1$ parameter vectors $b$ and $c$.

The excess demand function is,

$$
e_i(p) = q_i^d(p) - q_i^s(p), \quad i = 0, 1
$$

An equilibrium price vector $p^*$ satisfies $e_i(p^*) = 0$.

We set

$$
A = \begin{pmatrix}
            a_{00} & a_{01} \\
            a_{10} & a_{11}
        \end{pmatrix},
            \qquad 
    b = \begin{pmatrix}
            b_0 \\
            b_1
        \end{pmatrix}
    \qquad \text{and} \qquad
    c = \begin{pmatrix}
            c_0 \\
            c_1
        \end{pmatrix}
$$

for this particular question.


### The Multivariable Market Equilibrium

We can now easily get the multivariable version of the problem above.

The supply function remain unchanged,

$$ 
q^s (\bf{p}) =\bf{b} \bf\sqrt{p} 
$$

The demand function is,

$$ 
q^d (\bf{p}) = \text{exp}(- \bf{A} \cdot \bf{p}) + \bf{c}
$$

Our new excess demand function is,

$$
e(\bf{p}) = \text{exp}(- \bf{A} \cdot \bf{p}) + \bf{c} - \bf{b} \bf\sqrt{p} 
$$

The function below calculates the excess demand for the given parameters

```{code-cell} python3
def e(p, A, b, c):
    return jnp.exp(- A @ p) + c - b * jnp.srqt(p)
```


### Using Newton's Method

Now let's use the multivariate version of Newton's method to compute the equilibrium price

```{math}
:label: multi-newton
p_{n+1} = p_n - J_e(p_n)^{-1} e(p_n)
```

<!-- This is a multivariate version of [](https://python.quantecon.org/newton_method.html#oneD-newton) -->

Here $J_e(p_n)$ is the Jacobian of $e$ evaluated at $p_n$.

The iteration starts from some initial guess of the price vector $p_0$. 

Here, instead of coding Jacobian by hand, We use the `jax.jacobian()` function to auto-differentiate and calculate the Jacobian.

With only slight modification, we can generalize [our previous attempt](https://python.quantecon.org/newton_method.html#first_newton_attempt) to multi-dimensional problems

```{code-cell} python3
def newton(f, x_0, tol=1e-5, max_iter=10):
    x = x_0
    q = jax.jit(lambda x: x - jnp.linalg.solve(jax.jacobian(f)(x), f(x)))
    error = tol + 1
    n = 0
    while error > tol:
        n+=1
        if(n > max_iter):
            raise Exception('Max iteration reached without convergence')
        y = q(x)
        if(any(jnp.isnan(y))):
            raise Exception('Solution not found with NaN generated')
        error = jnp.linalg.norm(x - y)
        x = y
        print(f'iteration {n}, error = {error:.5f}')
    print('\n' + f'Result = {x} \n')
    return x
```

<!-- We find the algorithm terminates in 4 steps

```{code-cell} python3
%%time
p = newton(lambda p: e(p, A, b, c), init_p).block_until_ready()
```

```{code-cell} python3
jnp.max(jnp.abs(e(p, A, b, c)))
``` -->

<!-- The result is very accurate.  -->

<!-- With the larger overhead, the speed is not better than the optimized `scipy` function.

However, things will change when we move to higher dimensional problems. -->




### A High-Dimensional Problem

We now apply the multivariate Newton's Method to  investigate a large market with 5,000 goods.

We randomly generate the matrix $A$ and set the parameter vectors $b \text{ and } c$ to $\bf{1}$.


```{code-cell} python3
dim = 5_000
np.random.seed(123)
# Create a random matrix A and normalize the rows to sum to one
A = np.random.rand(dim, dim)
A = jnp.asarray(A)
s = jnp.sum(A, axis=0)
A = A / s
# Set up b and c
b = jnp.ones(dim)
c = jnp.ones(dim)
```

<!-- Here is essentially the same demand function we applied before, but now using `jax.numpy` for the calculations.

```{code-cell} python3
def e(p, A, b, c):
    return jnp.exp(- jnp.dot(A, p)) + c - b * jnp.sqrt(p)
``` -->

Here's our initial condition $p_0$

```{code-cell} python3
init_p = jnp.ones(dim)
```

By leveraging the power of Newton's method, JAX accelerated linear algebra,
automatic differentiation, and a GPU, we obtain a relatively small error for
this very large problem in just a few seconds:

```{code-cell} python3
%%time
p = newton(lambda p: e(p, A, b, c), init_p).block_until_ready()
```

```{code-cell} python3
np.max(np.abs(e(p, A, b, c)))
```

With the same tolerance, SciPy's `root` function takes much longer to run,
even with the Jacobian supplied.


```{code-cell} python3
%%time
solution = root(lambda p: e(p, A, b, c),
                init_p, 
                jac=lambda p: jax.jacobian(e)(p, A, b, c), 
                method='hybr',
                tol=1e-5)
```

```{code-cell} python3
p = solution.x
np.max(np.abs(e(p, A, b, c)))
```

The result is also less accurate.



## Exercises

```{exercise-start}
:label: newton_ex1
```

Consider a three-dimensional extension of the Solow fixed point problem with

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
The result should converge to the [analytical solution](solved_k).
````

```{exercise-end}
```


```{solution-start} newton_ex1
:class: dropdown
```

Let's first define the parameters for this problem

```{code-cell} python3
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

Then define the multivariate version of the formula for the [law of motion of captial](motion_law)

```{code-cell} python3
def multivariate_solow(k, A=A, s=s, α=α, δ=δ):
    return (s * jnp.dot(A, k**α) + (1 - δ) * k)
```

Let's run through each starting value and see the output

```{code-cell} python3
attempt = 1
for init in initLs:
    print(f'Attempt {attempt}: Starting value is {init} \n')
    %time k = newton(lambda k: multivariate_solow(k) - k, \
                     init).block_until_ready()
    print('-'*64)
    attempt +=1
```

We find that the results are invariant to the starting values given the well-defined property of this question.

But the number of iterations it takes to converge is dependent on the starting values.

Let substitute the output back to the formulate to check our last result

```{code-cell} python3
multivariate_solow(k) - k
```

Note the error is very small.

We can also test our results on the known solution

```{code-cell} python3
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

The result is very close to the ground truth but still slightly different.

We can increase the precision of the floating point numbers and restrict the tolerance to obtain a more accurate approximation (see detailed discussion in the [lecture on JAX](https://python-programming.quantecon.org/jax_intro.html#differences))

```{code-cell} python3
from jax.config import config
config.update("jax_enable_x64", True)
init = init.astype('float64')
%time k = newton(lambda k: multivariate_solow(k, A=A, s=s, α=α, δ=δ) - k, \
                 init,\
                 tol=1e-7).block_until_ready()
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

Set the tolerance to $0.0$ for more accurate output.


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

```{code-cell} python3
A = jnp.array([
    [0.2, 0.1, 0.7],
    [0.3, 0.2, 0.5],
    [0.1, 0.8, 0.1]
])
b = jnp.array([1.0, 1.0, 1.0])
c = jnp.array([1.0, 1.0, 1.0])
initLs = [jnp.repeat(5.0, 3),
          jnp.ones(3),
          jnp.array([4.5, 0.1, 4.0])] 
```

Let’s run through each initial guess and check the output

```{code-cell} python3
attempt = 1
for init in initLs:
    print(f'Attempt {attempt}: Starting value is {init} \n')
    init = init.astype('float64')
    %time p = newton(lambda p: e(p, A, b, c), \
                 init, \
                 tol=0.0).block_until_ready()
    print('-'*64)
    attempt +=1
```

We can find that Newton's method may fail for some starting values.

Sometimes it may take a few initial guesses to achieve convergence.

Substitute the result back to the formula to check our result

```{code-cell} python3
e(p, A, b, c)
```

We can see the result is very accurate.

```{solution-end}
```

