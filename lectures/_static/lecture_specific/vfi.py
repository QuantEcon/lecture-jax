# Implements VFI-Value Function iteration

def value_iteration(model, tol=1e-5):
    constants, sizes, arrays = model
    vz = jnp.zeros(sizes)
    _T = lambda v: T(v, constants, sizes, arrays)
    v_star = successive_approx_jax(_T, vz, tolerance=tol)
    return get_greedy(v_star, constants, sizes, arrays)
