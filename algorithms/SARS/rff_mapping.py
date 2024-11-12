import numpy as np
import torch


# Z(s)
# s (21,)  W (21,100)
def rff_mapping(s, M, sigma):
    s = np.array(s)
    k = len(s)  # 维度
    W = np.random.normal(0, 1 / sigma, (k, M))  # 随机矩阵 W
    b = np.random.uniform(0, 2 * np.pi, M)  # 随机偏移量 b
    return np.sqrt(2 / M) * np.cos(np.dot(W.T, s) + b)

    # gpu
    # s = torch.tensor(s, dtype=torch.float32, device='cuda')  # 将输入转为 GPU 张量
    #
    # k = len(s)  # 维度
    # W = torch.normal(0, 1 / sigma, (k, M), dtype=torch.float32, device='cuda')  # 随机矩阵 W
    # b = torch.empty(M, dtype=torch.float32, device='cuda').uniform_(0, 2 * np.pi)  # 随机偏移量 b
    #
    # return torch.sqrt(torch.tensor(2 / M, dtype=torch.float32, device='cuda')) * torch.cos(torch.matmul(W.T, s) + b)