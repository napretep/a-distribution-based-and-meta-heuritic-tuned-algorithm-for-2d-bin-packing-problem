# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'kde.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 5:19'
"""


import seaborn as sns
import matplotlib.pyplot as plt
from constant import 外包_data,COL,华为杯_data,随机_data

from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# # 创建一个由两个高斯分布混合而成的数据集
# data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(4, 0.8, 500)])
#
# # 为了能够使用 scikit-learn 的 KernelDensity，我们需要将数据转换成正确的形状
# data = data[:, np.newaxis]

# 创建 KernelDensity 对象并设置 Epanechnikov 核

# 在网格上计算 KDE
# density = np.exp(kde.score_samples(grid))  # 注意：score_samples 返回的是对数密度

def kde(data, bandwidth=0.02,count=200,kernel="gaussian"):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
    x = np.linspace(0,1, count)
    y = np.linspace(0,1, count)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    Z =np.exp(kde.score_samples(grid)).reshape(X.shape)
    return X,Y,Z

def draw_contour(pic_pos,data,text):
    ax = plt.subplot(*pic_pos)
    ax.set_aspect('equal')
    sns.kdeplot(x=data[0].ravel(), y=data[1].ravel(),weights=data[2].ravel(),fill=True)
    # c = ax.contourf(data[0], data[1], data[2], levels=20)
    # plt.colorbar(c, ax=ax)
    # ax.colorbar(label='Density')
    # plt.show()
    plt.title(text)
    # contour_filled = plt.contourf(*data, levels=10)  # 填充等高线




if __name__ == "__main__":

    pass
    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    draw_contour((1, 3, 1), kde(随机_data[:, COL.maxL:COL.minL + 1]),"random data")
    draw_contour((1, 3, 2), kde(外包_data[:,COL.maxL:COL.minL+1]),"wooden data")
    draw_contour((1, 3, 3), kde(华为杯_data[:, COL.maxL:COL.minL + 1]),"metal data")


    # draw_contour((1, 3, 1), 随机_data[:, COL.maxL:COL.minL + 1])
    # draw_contour((1, 3, 2), 外包_data[:,COL.maxL:COL.minL+1])
    # draw_contour((1, 3, 3),  华为杯_data[:, COL.maxL:COL.minL + 1])

    # 绘制等高线图
    # 使用 seaborn 的 heatmap 函数绘制结果
    # sns.heatmap(Z, cmap='viridis')
    plt.show()