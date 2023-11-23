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


def plot_hist2d(datas,titles):
    """数据的统计图"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


    for i in range(len(datas)):
        g= sns.jointplot(x=datas[i][:,1], y=datas[i][:,2], kind='hist',joint_kws={'bins': 50}, marginal_kws={'bins': 50},marginal_ticks=True)
        plt.subplots_adjust(bottom=0.2)
        # g.fig.save(f"test{time()}")
        g.fig.suptitle(titles[i], y=0.1)
        plt.savefig(f"test{time()}.png")



def plot_data_dist():
    plot_hist2d([data_sets[name] for name in data_sets.keys()],
                titles=[name for name in data_sets.keys()])
    plt.show()

def plot_kde_sampled_data_dist():
    plot_hist2d([kde_sample(data_sets[name],5000) for name in data_sets.keys()],
                titles=[f"kde sampled {name}" for name in data_sets.keys()])
    plt.show()

def plot_kde_noised_sample_data_dist():
    noised_ratio = [(0,(i+1)/10) for i in range(5)]
    data_names = list(data_sets.keys())
    noised_data = [random_mix(kde_sample(data_sets[data_names[i]], 5000)[:,1:],noised_ratio[i]) for i in range(len(data_names)) ]
    plot_hist2d(noised_data,titles=[f"noised sampled {data_names[i]} noised ratio {noised_ratio[i][1]}" for i in range(len(data_names))])
    plt.show()

if __name__ == "__main__":
    plot_data_dist()
    pass

