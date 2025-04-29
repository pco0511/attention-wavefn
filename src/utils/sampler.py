from functools import partial

import jax
import jax.numpy as jnp

def make_mh_sampler(indicator_fn, unnormed_pdf):
    
    @partial(jax.jit, static_nums=(2, 3, 4,))
    def metropolis_hastings(key, x0, num_steps, dim, sigma):
        # 랜덤 키 분할
        keys = jax.random.split(key, num_steps)

        # carry: (현재 위치 x, 지금까지 수락 횟수)
        def mh_step(carry, key):
            x, acc_count = carry
            key_prop, key_u = jax.random.split(key)

            # 제안 샘플
            x_prop = x + sigma * jax.random.normal(key_prop, (dim,))

            # 수락 확률 계산
            alpha = jnp.where(
                indicator_fn(x_prop) > 0,
                jnp.minimum(1.0, unnormed_pdf(x_prop) / unnormed_pdf(x)),
                0.0
            )
            u = jax.random.uniform(key_u)
            accept = u < alpha

            # 다음 상태 결정
            x_next = jnp.where(accept, x_prop, x)
            acc_count_next = acc_count + accept.astype(jnp.int32)

            return (x_next, acc_count_next), x_next

        # 초기 carry: 위치 x0, 수락 횟수 0
        (x_final, total_accept), samples = jax.lax.scan(
            mh_step,
            (x0, jnp.array(0, dtype=jnp.int32)),
            keys
        )

        # 수락률 계산
        acceptance_rate = total_accept / num_steps
        return samples, acceptance_rate

    return metropolis_hastings

@partial(jax.jit, static_argnums=(2, 3, 4, 5,))
def metropolis_hastings(key, x0, num_steps, sigma, indicator_fn, unnorm_fn):
    """
    단일 체인 Metropolis-Hastings 샘플러
    indicator_fn, unnorm_fn을 인자로 받음 (static_argnames 사용)
    """
    keys = jax.random.split(key, num_steps)

    def mh_step(carry, key):
        x, acc_count = carry
        key_prop, key_u = jax.random.split(key)

        x_prop = x + sigma * jax.random.normal(key_prop, x0.shape) # <<<< 밖으로 빼기

        alpha = jnp.where(
            indicator_fn(x_prop) > 0,
            jnp.minimum(1.0, unnorm_fn(x_prop) / unnorm_fn(x)),  # <<<< 계산된거 넘겨주기
            0.0
        )
        u = jax.random.uniform(key_u) # <<<< 밖으로 빼기
        accept = u < alpha

        x_next = jnp.where(accept, x_prop, x)
        acc_count_next = acc_count + accept.astype(jnp.int32)

        return (x_next, acc_count_next), x_next

    (x_final, total_accept), samples = jax.lax.scan(
        mh_step,
        (x0, jnp.array(0, dtype=jnp.int32)),
        keys
    )

    acceptance_rate = total_accept / num_steps
    return samples, acceptance_rate