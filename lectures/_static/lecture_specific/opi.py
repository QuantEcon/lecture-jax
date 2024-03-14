def optimistic_policy_iteration(model, tol=1e-5, m=10):
    """
    Implements optimistic policy iteration (see dp.quantecon.org)
    """
    params, sizes, arrays = model
    v = jnp.zeros(sizes)
    error = tol + 1
    while error > tol:
        last_v = v
        σ = get_greedy(v, params, sizes, arrays)
        for _ in range(m):
            v = T_σ(v, σ, params, sizes, arrays)
        error = jnp.max(jnp.abs(v - last_v))
    return get_greedy(v, params, sizes, arrays)
