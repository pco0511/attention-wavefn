from functools import partial

import jax
import jax.numpy as jnp

import equinox as eqx

# def make_mh_sampler(indicator_fn, unnormed_pdf):
    
#     @partial(jax.jit, static_nums=(2, 3, 4,))
#     def metropolis_hastings(key, x0, num_steps, dim, sigma):
#         # 랜덤 키 분할
#         keys = jax.random.split(key, num_steps)
        
        
#         # carry: (현재 위치 x, 지금까지 수락 횟수)
#         def mh_step(carry, key):
#             x, acc_count = carry
#             key_prop, key_u = jax.random.split(key)

#             # 제안 샘플
#             x_prop = x + sigma * jax.random.normal(key_prop, (dim,))

#             # 수락 확률 계산
#             alpha = jnp.where(
#                 indicator_fn(x_prop) > 0,
#                 jnp.minimum(1.0, unnormed_pdf(x_prop) / unnormed_pdf(x)),
#                 0.0
#             )
#             u = jax.random.uniform(key_u)
#             accept = u < alpha

#             # 다음 상태 결정
#             x_next = jnp.where(accept, x_prop, x)
#             acc_count_next = acc_count + accept.astype(jnp.int32)

#             return (x_next, acc_count_next), x_next

#         # 초기 carry: 위치 x0, 수락 횟수 0
#         (x_final, total_accept), samples = jax.lax.scan(
#             mh_step,
#             (x0, jnp.array(0, dtype=jnp.int32)),
#             keys
#         )

#         # 수락률 계산
#         acceptance_rate = total_accept / num_steps
#         return samples, acceptance_rate

#     return metropolis_hastings

@eqx.filter_jit
def metropolis_hastings(key, x0, num_steps, sigma, indicator_fn, unnorm_fn): # < weight 만 넘겨줘서 매번 jit안하게 하기
    def mh_step(carry, rv):
        x, p, acc_count = carry
        u, prop_noise = rv

        x_prop = x + sigma * prop_noise
        p_prop = unnorm_fn(x_prop)
        
        alpha = jnp.where(
            indicator_fn(x_prop) > 0,
            jnp.minimum(1.0, p_prop / p),
            0.0
        )
        
        accept = u < alpha
        x_next = jnp.where(accept, x_prop, x)
        p_next = jnp.where(accept, p_prop, p)
        acc_count_next = acc_count + accept.astype(jnp.int32)

        return (x_next, p_next, acc_count_next), x_next
    
    key_u, key_prop = jax.random.split(key, 2)
    us = jax.random.uniform(key_u, (num_steps))
    props = jax.random.normal(key_prop, (num_steps, *x0.shape)) 
    
    p0 = unnorm_fn(x0)
    
    (_, _, total_accept), samples = jax.lax.scan(
        mh_step,
        (x0, p0, jnp.array(0, dtype=jnp.int32)),
        (us, props)
    )

    acceptance_rate = total_accept / num_steps
    return samples, acceptance_rate