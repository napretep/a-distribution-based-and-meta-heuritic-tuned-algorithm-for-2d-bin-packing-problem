# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'test3.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/8 0:44'
"""

from constant import *
import BinPacking2DAlgo


import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":

    pass
    # 假设你已经有了一些数据
    noise_levels = np.arange(0, 31, 1) # 噪声级别从0%到30%
    # 假设算法在各个噪声级别上的结果是一个随机过程（你需要用你自己的数据替换这部分）
    algorithm1_result = np.random.normal(loc=0.5, scale=0.1, size=(10, len(noise_levels)))
    algorithm2_result = np.random.normal(loc=0.5, scale=0.2, size=(10, len(noise_levels)))

    # 计算平均值和标准差
    algorithm1_mean = np.mean(algorithm1_result, axis=0)
    algorithm1_std = np.std(algorithm1_result, axis=0)
    algorithm2_mean = np.mean(algorithm2_result, axis=0)
    algorithm2_std = np.std(algorithm2_result, axis=0)

    # 创建一个新的图形
    plt.figure(figsize=(8, 6))

    # 创建折线图，并添加误差条
    plt.errorbar(noise_levels, algorithm1_mean, yerr=algorithm1_std, label='Algorithm 1', alpha=0.6)
    plt.errorbar(noise_levels, algorithm2_mean, yerr=algorithm2_std, label='Algorithm 2', alpha=0.6)

    # 添加标题和标签
    plt.title('Algorithm Performance vs. Noise Level')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Algorithm Performance')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()