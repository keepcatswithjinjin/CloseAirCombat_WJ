import numpy as np


def log_reward_mapping(rs_i, R_min=1, R_max=10, k=1):
    """
    计算奖励值 R^S(s_i)，基于成功率 x 的对数映射。

    参数:
    x (float): 成功率，范围 [0, 1]。
    R_min (float): 奖励最小值。
    R_max (float): 奖励最大值。
    k (float): 对数增长速率调节系数，控制对数映射的曲线陡峭度。

    返回:
    float: 映射后的奖励值 R^S(s_i)。
    """
    # 确保成功率在 [0, 1] 范围内
    rs_i = np.clip(rs_i, 0, 1)
    # 计算对数映射后的奖励值
    rs = R_min + (R_max - R_min) * np.log(1 + k * rs_i)

    return rs
