import numpy as np

def gaussian_kernel_density(s_i, buffer, sigma):   # s_i 本次obs
    # buffer: 成功或者失败状态存储的数组
    density = 0
    for s_j in buffer:
        density += np.exp(-np.linalg.norm(s_i - s_j)**2 / (2 * sigma**2))
    return density / len(buffer)
