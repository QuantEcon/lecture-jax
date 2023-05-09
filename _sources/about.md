
# About

This lecture series provides an introduction to quantitative economics using [Google JAX](https://github.com/google/jax).


## What is JAX?

JAX is an open source Python library developed by Google Research to support
in-house artificial intelligence and machine learning.

JAX provides data types, functions and a compiler for fast linear
algebra operations and automatic differentiation.

Loosely speaking, JAX is like [NumPy](https://numpy.org/) with the addition of

* automatic differentiation
* automated GPU/TPU support
* a just-in-time compiler

One of the great benefits of JAX is that exactly the same code can be run either
on the CPU or on a hardware accelerator, such as a GPU or TPU.

In short, JAX delivers

1. high execution speeds on CPUs due to efficient parallelization and JIT
   compilation,
1. a powerful and convenient environment for GPU programming, and
1. the ability to efficiently differentiate smooth functions for optimization
   and estimation.

These features make JAX ideal for almost all quantitative economic modeling
problems that require heavy-duty computing.

## How to run these lectures

The easiest way to run these lectures is via  [Google Colab](https://colab.research.google.com/).

JAX is pre-installed with GPU support on Colab and Colab provides GPU access
even on the free tier.

Each lecture has a "play" button on the top right that you can use to launch the
lecture on Colab.

You might also like to try using JAX locally.

If you do not own a GPU, you can still install JAX for the CPU by following the relevant [install instructions](https://github.com/google/jax).

(We recommend that you install [Anaconda
Python](https://www.anaconda.com/download) first.)

If you do have a GPU, you can try installing JAX for the GPU by following the
install instructions for GPUs.

(This is not always trivial but is starting to get easier.)

## Credits

In building this lecture series, we had invaluable assistance from research
assistants at QuantEcon and our QuantEcon colleagues.

In particular, we thank and credit 

- [Shu Hu](https://github.com/shlff)
- [Smit Lunagariya](https://github.com/Smit-create)
- [Matthew McKay](https://github.com/mmcky)
- [Humphrey Yang](https://github.com/HumphreyYang)
- [Hengcheng Zhang](https://github.com/HengchengZhang)
- [Frank Wu](https://github.com/chappiewuzefan)


## Prerequisites

We assume that readers have covered most of the QuantEcon lecture
series [on Python programming](https://python-programming.quantecon.org/intro.html).  

