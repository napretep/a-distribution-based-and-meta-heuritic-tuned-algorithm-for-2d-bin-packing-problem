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
import numpy as np
import matplotlib.pyplot as plt


def plot_hist2d(sj_data, hw_data, wb_data,titles):
    """数据的统计图"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    data = [sj_data, hw_data, wb_data]

    for i in range(3):
        g= sns.jointplot(x=data[i][:,1], y=data[i][:,2], kind='hist',joint_kws={'bins': 50}, marginal_kws={'bins': 50},marginal_ticks=True)
        plt.subplots_adjust(bottom=0.2)
        g.fig.suptitle(titles[i], y=0.1)


def plot_data_dist():
    plot_hist2d(随机_data, 华为杯_data, 外包_data,
                titles=[RANDOMGEN_DATA, PRODUCTION_DATA1, PRODUCTION_DATA2])
    plt.show()

def plot_kde_sampled_data_dist():
    plot_hist2d(kde_sample(随机_data, 5000), kde_sample(华为杯_data, 5000), kde_sample(外包_data, 5000),
                titles=[f"kde sampled {RANDOMGEN_DATA}", f"kde sampled {PRODUCTION_DATA1}",f"kde sampled {PRODUCTION_DATA2}",])
    plt.show()

if __name__ == "__main__":
    plot_kde_sampled_data_dist()
    pass

