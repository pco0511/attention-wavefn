import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any
from jaxtyping import Bool, Int, Float, Complex, Array, PRNGKeyArray

LogPsiFN = Callable[[Float[Array, "n_par spc_dim"]], Complex[Array, ""]]
IndicatorFN = Callable[[Float[Array, "n_par spc_dim"]], bool]

class MCMCState(eqx.Module):
    points: Float[Array, "n_chn n_par spc_dim"]
    log_psi_real: Float[Array, "n_chn"]
    key: PRNGKeyArray
    accepted_count: Int[Array, "n_chn"]
    samples_collected_count: Int[Array, ""]
    samples_buffer: Float[Array, "samp_per_chn n_chn n_par spc_dim"]
    
def _mcmc_step(
    state: MCMCState,
    step_idx: int,
    log_psi_module: eqx.Module,
    indicator_fn: IndicatorFN,
    sigma: float,
    n_burn_in: int,
    thinning: int,
    n_particles: int,
    space_dim: int
) -> tuple[MCMCState, tuple[Float[Array, "n_chn n_par spc_dim"], Int[Array, "n_chn"]]]:
    """
    Performs a single MCMC step for all chains in parallel.
    This function will be used inside jax.lax.scan.
    """
    key, proposal_key, accept_key = jax.random.split(state.key, 3)
    
    # propose new points
    # noise shape: (n_chains, n_particles, space_dim)
    noise = jax.random.normal(proposal_key, state.points.shape) * sigma
    proposed_points = state.points + noise
    
    # check_indicator
    vmap_indicator_fn_local = jax.vmap(indicator_fn, in_axes=0, out_axes=0)
    is_valid_proposal = vmap_indicator_fn_local(proposed_points) # (n_chains,)
    
    vmap_log_psi_module_local = jax.vmap(log_psi_module, in_axes=0, out_axes=0)
    proposed_log_psi_complex = vmap_log_psi_module_local(proposed_points) # (n_chains,)
    proposed_log_psi_real = jnp.real(proposed_log_psi_complex)
    
    # If proposal is invalid, treat as if log_psi is -inf (reject)
    proposed_log_psi_real = jnp.where(
        is_valid_proposal,
        proposed_log_psi_real,
        -jnp.inf
    )
    
    # calculate acceptance probability
    # P_accept = min(1, |psi_new/psi_old|^2)
    # log(P_accept) = min(0, 2 * (log_psi_new_real - log_psi_old_real))
    log_acceptance_ratio = 2.0 * (proposed_log_psi_real - state.log_psi_real)
    
    
    # Numerical stability: accept if ratio > log(uniform)
    # Equivalent to uniform < exp(log_acceptance_ratio)
    # Ensure can_accept is true if proposed_log_psi_real is not -inf
    can_accept = proposed_log_psi_real > -jnp.inf # (n_chains,)
    uniform_draw = jax.random.uniform(accept_key, shape=(state.points.shape[0],)) # (n_chains,)
    
    accept_flags = (jnp.log(uniform_draw) < log_acceptance_ratio) & can_accept # (n_chains,)

    # Update positions and log_psi based on acceptance
    next_points = jnp.where(
        accept_flags[:, None, None], # Expand accept_flags to match position dimensions
        proposed_points,
        state.points
    )
    next_log_psi_real = jnp.where(
        accept_flags,
        proposed_log_psi_real,
        state.log_psi_real
    )
    
    new_accepted_count = state.accepted_count + accept_flags.astype(jnp.int32)
    
    # Store samples after burn-in and at thinning intervals
    # Check if this step should be recorded
    current_sample_idx = (step_idx - n_burn_in) // thinning
    should_collect_sample = (step_idx >= n_burn_in) & ((step_idx - n_burn_in) % thinning == 0) & (current_sample_idx < state.samples_buffer.shape[0])

    # Update samples_buffer - use lax.dynamic_update_slice for one chain at a time inside vmap or carefully index
    # For simplicity, we can update the whole slice.
    # Ensure that current_sample_idx is valid before updating.
    updated_samples_buffer = jax.lax.cond(
        should_collect_sample,
        lambda: state.samples_buffer.at[current_sample_idx].set(next_points),
        lambda: state.samples_buffer
    )
    new_samples_collected_count = state.samples_collected_count + jnp.asarray(should_collect_sample, dtype=jnp.int32)

    next_state = MCMCState(
        points=next_points,
        log_psi_real=next_log_psi_real,
        key=key,
        accepted_count=new_accepted_count,
        samples_collected_count=new_samples_collected_count,
        samples_buffer=updated_samples_buffer
    )
    
    # `scan` y_carry: what to collect at each step
    # Here we collect current positions and acceptance flags for potential diagnostics
    # However, the primary samples are stored in `state.samples_buffer`
    return next_state, (next_points, accept_flags)
    
@eqx.filter_jit
def metropolis_hastings_sampler(
    key: PRNGKeyArray,
    initial_points: Float[Array, "n_chn n_par spc_dim"], # (n_chains, n_particles, space_dim)
    log_psi_module: eqx.Module,   # Equinox module, takes (n_particles, space_dim)
    indicator_fn: IndicatorFN,    # Takes (n_particles, space_dim)
    n_target_samples: int,        # Total samples to output: n_batch
    n_chains: int,
    n_burn_in_per_chain: int,
    thinning: int,
    sigma: float
) -> tuple[Float[Array, "n_batch spc_dim"], Float[Array, ""], MCMCState]: # (samples, acceptance_rate, final_mcmc_state)
    """
    Metropolis-Hastings sampler for VMC using JAX and Equinox.

    Args:
        key: JAX PRNG key.
        initial_points: Starting electron configurations for each chain.
                           Shape (n_chains, n_particles, space_dim).
        log_psi_module: An Equinox Module that computes log(Psi(R)).
                        It should take (n_particles, space_dim) and return a complex scalar.
        indicator_fn: A function that takes (n_particles, space_dim) and returns True if
                      the configuration is valid (e.g., within bounds), False otherwise.
        n_target_samples: Total number of samples to generate and return (n_batch).
                          Must be a multiple of n_chains.
        n_chains: Number of parallel MCMC chains.
        n_burn_in_per_chain: Number of burn-in steps for each chain.
        thinning: Thinning factor. Collect one sample every `thinning` steps after burn-in.
        sigma: Standard deviation for the Gaussian proposal distribution.

    Returns:
        A tuple containing:
        - samples: Collected samples, shape (n_target_samples, n_particles, space_dim).
        - overall_acceptance_rate: Scalar acceptance rate across all chains and steps (after burn-in).
        - final_mcmc_state: The MCMCState object at the end of sampling.
    """
    if n_target_samples % n_chains != 0:
        raise ValueError("n_target_samples (n_batch) must be a multiple of n_chains.")

    n_samples_per_chain = n_target_samples // n_chains
    n_particles = initial_points.shape[1]
    space_dim = initial_points.shape[2]

    # Total MCMC steps per chain needed to collect n_samples_per_chain
    total_mcmc_steps_per_chain = n_burn_in_per_chain + n_samples_per_chain * thinning

    # Initial log_psi values
    # vmap log_psi_module over the chains dimension
    vmap_log_psi_module_init = jax.vmap(log_psi_module, in_axes=0, out_axes=0)
    initial_log_psi_complex = vmap_log_psi_module_init(initial_points) # (n_chains,)
    initial_log_psi_real = jnp.real(initial_log_psi_complex)

    # Initial MCMC state
    initial_state = MCMCState(
        points=initial_points,
        log_psi_real=initial_log_psi_real,
        key=key,
        accepted_count=jnp.zeros(n_chains, dtype=jnp.int32),
        samples_collected_count=jnp.array(0, dtype=jnp.int32),
        samples_buffer=jnp.zeros((n_samples_per_chain, n_chains, n_particles, space_dim), dtype=initial_points.dtype)
    )
    
    # Curry the static arguments for _mcmc_step
    # The `log_psi_module` and `indicator_fn` are passed as closures.
    # JAX's JIT will handle Equinox modules by tracing their array parts (parameters)
    # and treating their static parts (like Python functions, static fields) as static.
    # We need to ensure these are treated as static for the scan step function.
    # One way is to make them part of the carry state if they change, or pass them as static args to scan if possible.
    # Here, they are arguments to `curried_step_fn`, so JAX's JIT will treat them appropriately.
    # JAX captures these variables by reference when `curried_step_fn` is defined.
    
    # Wrap log_psi_module and indicator_fn with partial if they have their own parameters
    # or ensure they are properly handled by JIT (Equinox modules usually are)
    # The _mcmc_step function already takes them as arguments.

    def curried_step_fn(carry_state, step_idx):
        # log_psi_module and indicator_fn are closed over from the outer scope
        return _mcmc_step(
            carry_state, step_idx,
            log_psi_module, indicator_fn, # These are "static" for the scan
            sigma, n_burn_in_per_chain, thinning,
            n_particles, space_dim
        )

    # Run the MCMC simulation using lax.scan
    final_state, scan_outputs = jax.lax.scan(
        curried_step_fn,
        initial_state,
        jnp.arange(total_mcmc_steps_per_chain)
    )
    # scan_outputs is a tuple: (all_next_positions_over_steps, all_acceptance_flags_over_steps)
    # Each element in scan_outputs has shape (total_mcmc_steps_per_chain, n_chains, ...)

    # Extract collected samples from the buffer
    # samples_buffer has shape (n_samples_per_chain, n_chains, n_particles, space_dim)
    # We need to reshape it to (n_target_samples, n_particles, space_dim)
    # (n_samples_per_chain * n_chains, n_particles, space_dim)
    collected_samples = final_state.samples_buffer # This already has the collected samples
    
    # Reshape: transpose and then reshape
    # (n_chains, n_samples_per_chain, n_particles, space_dim)
    samples_reshaped = jnp.transpose(collected_samples, (1, 0, 2, 3))
    # (n_chains * n_samples_per_chain, n_particles, space_dim)
    samples_final = jnp.reshape(samples_reshaped, (n_target_samples, n_particles, space_dim))
    
    # Calculate acceptance rate (only for steps after burn-in)
    # scan_outputs[1] contains acceptance_flags with shape (total_mcmc_steps_per_chain, n_chains)
    acceptance_flags_all_steps = scan_outputs[1]
    acceptance_flags_after_burn_in = acceptance_flags_all_steps[n_burn_in_per_chain:]
    
    total_proposals_after_burn_in = (total_mcmc_steps_per_chain - n_burn_in_per_chain) * n_chains
    total_accepted_after_burn_in = jnp.sum(acceptance_flags_after_burn_in) # Sums over all steps and chains
    
    overall_acceptance_rate = jnp.asarray(total_accepted_after_burn_in, dtype=jnp.float32) / jnp.asarray(total_proposals_after_burn_in, dtype=jnp.float32)
    # A slightly different way to get accepted_count for post-burn-in from final_state.accepted_count
    # would require tracking accepted_count at burn-in point. The current sum is more direct.


    # As a sanity check, ensure all requested samples were collected
    #jax.debug.print("Samples collected count: {}", final_state.samples_collected_count)
    #jax.debug.print("Expected samples per chain: {}", n_samples_per_chain)

    return samples_final, overall_acceptance_rate, final_state
    