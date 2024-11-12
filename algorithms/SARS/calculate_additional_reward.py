from concurrent.futures import ThreadPoolExecutor, as_completed
from algorithms.SARS.rff_kernel_density import rff_kernel_density
import numpy as np


def calculate_additional_reward(env_idx, agent_idx, obs, Ds_buffer, Df_buffer, M, sigma):
    # Ns_i_density 和 Nf_i_density 的计算
    Ns_i_density = rff_kernel_density(obs, Ds_buffer[env_idx][agent_idx], M, sigma) if Ds_buffer[env_idx][
        agent_idx] else 0
    Nf_i_density = rff_kernel_density(obs, Df_buffer[env_idx][agent_idx], M, sigma) if Df_buffer[env_idx][
        agent_idx] else 0
    total_success = len(Ds_buffer[env_idx][agent_idx])
    total_failure = len(Df_buffer[env_idx][agent_idx])
    N = total_success + total_failure
    Ns_i_count = N * Ns_i_density
    Nf_i_count = N * Nf_i_density
    alpha = Ns_i_count + 1
    beta = Nf_i_count + 1
    return np.random.beta(alpha, beta)
