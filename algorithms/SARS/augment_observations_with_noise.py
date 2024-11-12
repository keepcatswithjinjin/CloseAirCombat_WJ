import numpy as np


def augment_observations_with_noise(observations, noise_std=0.01, num_copies=30):
    """
    为轨迹中的每个观察值添加噪声，生成多条增强轨迹。

    参数:
    - observations: 原始轨迹中的观察值列表，每个元素是一个观察值（例如numpy数组）
    - noise_std: 噪声的标准差，控制噪声幅度
    - num_copies: 生成的增强轨迹数量

    返回:
    - augmented_trajectories: 增强后的轨迹列表，每条轨迹是一个带噪声的观察值序列
    """
    augmented_trajectories = []

    for _ in range(num_copies):
        noisy_trajectory = [
            obs + np.random.normal(0, noise_std, size=np.shape(obs)) for obs in observations
        ]
        augmented_trajectories.append(noisy_trajectory)

    return augmented_trajectories
