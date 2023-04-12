---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

 ```{raw} html
 <div id="qe-notebook-header" align="right" style="text-align:right;">
         <a href="https://quantecon.org/" title="quantecon.org">
                 <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
         </a>
 </div>
 ```

# Using Newton’s Method to Solve Economic Models

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np
```

```{code-cell} ipython3
def e(p, A, b, c):
    return jnp.exp(- A @ p) + c - b * jnp.sqrt(p)

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

init_p = jnp.ones(dim)
```

```{code-cell} ipython3
%%time
p = newton(lambda p: e(p, A, b, c), init_p).block_until_ready()
```

```{code-cell} ipython3
jnp.max(jnp.abs(e(p, A, b, c)))
```