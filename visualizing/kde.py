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
from constant import *
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# def kde(data, bandwidth=0.05,count=500,kernel="epanechnikov"): #epanechnikov,gaussian
#     kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data[:,1:3])
#     x = np.linspace(0,1, count)
#     y = np.linspace(0,1, count)
#     X, Y = np.meshgrid(x, y)
#     grid = np.vstack([X.ravel(), Y.ravel()]).T
#     Z =np.exp(kde.score_samples(grid)).reshape(X.shape)
#     return X,Y,Z



def draw_contour(pic_pos,data,text):
    ax = plt.subplot(*pic_pos)
    ax.set_aspect('equal')
    Z=data[2].ravel()
    if np.max(Z) > 1:
        Z=Z/MATERIAL_SIZE[1]
    sns.kdeplot(x=data[0].ravel(), y=data[1].ravel(),weights=Z,fill=True)
    plt.title(text)


def plot_contour(sj_data, hw_data, wb_data):
    # plt.figure(figsize=(6,6))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    fig, axs = plt.subplots(1,3)  # 创建一个 subplot
    data = [sj_data, hw_data, wb_data]
    title = ["随机数据", "华为杯数据", "外包数据"]
    for i in range(3):
        ax = axs[i]
        # data_x = data[i][:1]*MATERIAL_SIZE[0]
        # data_y = data[i][:1]*MATERIAL_SIZE[1]
        cf = ax.contourf(*data[i], levels=10, cmap='viridis')  # 将输出存储为变量 `cf`
        fig.colorbar(cf, ax=ax)  # 添加颜色图例
        ax.axis('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title[i])
    # plt.show()

# 假设你已经使用你的 `kde` 函数生成了 X, Y, Z
def plot_hist2d(sj_data, hw_data, wb_data):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    fig, axs = plt.subplots(1, 3)  # 创建一个 subplot
    data = [sj_data, hw_data, wb_data]
    title = ["随机数据", "华为杯数据", "外包数据"]
    for i in range(3):
        ax = axs[i]
        sns.jointplot(x=data[i][:,1], y=data[i][:,2], kind='hist',joint_kws={'bins': 50}, marginal_kws={'bins': 50},)
        # sns.jointplot(x=data[i][:,0], y=data[i][:,1], kind='kde')

        ax.axis('equal')
        ax.set_title(title[i])

if __name__ == "__main__":
    plot_hist2d(kde_sample((随机_data)),kde_sample((华为杯_data)),kde_sample((外包_data)))
    # plot_contour(kde(unify(随机_data)), kde(unify(华为杯_data)), kde(unify(random_choice(外包_data))))
    # data = (华为杯_data)
    # plt.hist2d(data[:,1], data[:,2], bins=(100, 100), cmap="viridis")

    # 添加颜色条
    # plt.colorbar()

    # 设置坐标轴标签
    # plt.xlabel('Width')
    # plt.ylabel('Length')

    # 显示图形
    # plt.show()
    plt.show()