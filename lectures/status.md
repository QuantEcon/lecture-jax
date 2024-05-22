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

# Execution Statistics

This table contains the latest execution statistics.

```{nb-exec-table}
```

(status:machine-details)=

These lectures are built on `linux` instances through `github actions` that has 
access to a `gpu`. These lectures make use of the nvidia `T4` card.

You can check the backend used by JAX using:

```{code-cell} ipython3
import jax
# Check if JAX is using GPU
print(f"JAX backend: {jax.devices()[0].platform}")
```

and the hardware we are running on:

```{code-cell} ipython3
!nvidia-smi
```