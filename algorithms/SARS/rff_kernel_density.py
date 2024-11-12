import numpy as np
import torch

from algorithms.SARS.rff_mapping import rff_mapping


def rff_kernel_density(s_i, buffer, M, sigma):
    density = 0
    s_i = np.array(s_i)
    s_i = s_i.flatten()
    z_si = np.abs(rff_mapping(s_i, M, sigma))
    for s_j in buffer:
        z_sj = np.abs(rff_mapping(s_j, M, sigma))
        density += np.dot(z_si, z_sj)
    if len(buffer) == 0:
        return 0
    return density / len(buffer)

    # gpu
    # density = 0
    # s_i = torch.tensor(s_i, device='cuda').flatten()  # 将输入转为 GPU 张量并展平
    # z_si = torch.abs(rff_mapping(s_i, M, sigma))
    #
    # for s_j in buffer:
    #     z_sj = torch.abs(rff_mapping(torch.tensor(s_j, device='cuda'), M, sigma))  # 确保 s_j 也转为 GPU 张量
    #     density += torch.dot(z_si, z_sj)
    # if len(buffer) == 0:
    #     return 0
    # return (density / len(buffer)).cpu()


"""
    1.截断负值  当为负值时  使用较小的数值减小对beta分布的影响
    2.调整RFF参数  使得不出现负值
    3. 绝对值  保留相对密度关系 
"""
