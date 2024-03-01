def successive_approx_jax(x_0,                   # Initial condition
                          constants,
                          sizes,
                          arrays,                 
                          tolerance=1e-6,        # Error tolerance
                          max_iter=10_000):      # Max iteration bound

    def body_fun(k_x_err):
        k, x, error = k_x_err
        x_new = T(x, constants, sizes, arrays)
        error = jnp.max(jnp.abs(x_new - x))
        return k + 1, x_new, error

    def cond_fun(k_x_err):
        k, x, error = k_x_err
        return jnp.logical_and(error > tolerance, k < max_iter)

    k, x, error = jax.lax.while_loop(cond_fun, body_fun, (1, x_0, tolerance + 1))
    return x

successive_approx_jax = jax.jit(successive_approx_jax, static_argnums=(2,))
