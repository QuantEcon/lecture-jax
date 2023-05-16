
# About

Perhaps the single most notable feature of scientific computing in the past
decades is the rise and rise of parallel computation.

For example, the advanced artificial intelligence applications now shaking the
world of business and academia require massive computer power to train, and the
great majority of that computer power is supplied by GPUs.

For us economists, with our ever-growing need for more compute cycles,
parallel computing provides both opportunities and new difficulties.

The main difficulty we face vis-a-vis parallel computation is accessibility.

Even for those with time to invest in careful parallelization of their programs,
exploiting the full power of parallel hardware is challenging for non-experts.

Moreover, that hardware changes from year to year, so any human capital
associated with mastering intricacies of a particular GPU has a very high
depreciation rate.

For these reasons, we view [Google JAX](https://github.com/google/jax) as one of
the most exciting advances in scientific computing in recent years.

JAX makes high performance and parallel computing accessible. 

It provides a familiar array programming interface based on NumPy, and, as long as
some simple conventions are adhered to, this code compiles to extremely
efficient and well-parallelized machine code.

One of the most agreeable features of JAX is that the same code set and be run on
either CPUs or GPUs, which allows users to test and develop locally, before
deploying to a more powerful machine for heavier computations.

JAX is relatively easy to learn and highly portable, allowing us programmers to
focus on the algorithms we want to implement, rather than particular features of
our hardware.

This lecture series provides an introduction to using Google JAX for
quantitative economics.

The rest of this page provides some background information on JAX, notes on
how to run the lectures, and credits for our colleagues and RAs.



## What is JAX?

JAX is an open source Python library developed by Google Research to support
in-house artificial intelligence and machine learning.

JAX provides data types, functions and a compiler for fast linear
algebra operations and automatic differentiation.

Loosely speaking, JAX is like [NumPy](https://numpy.org/) with the addition of

* automatic differentiation
* automated GPU/TPU support
* a just-in-time compiler

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

