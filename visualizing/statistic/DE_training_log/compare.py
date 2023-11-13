# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 5:57'
"""
import os

import matplotlib.pyplot as plt

from constant import *


def run():
    for algo_name,gencount in [(AlgoName.Dist_Skyline,100),(AlgoName.Dist_MaxRect,500)]:
        for data_mode in [STANDARD, NOISED]:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # 创建一个1行3列的子图
            for idx, data_name in enumerate([PRODUCTION_DATA1, PRODUCTION_DATA2, RANDOMGEN_DATA]):
                training_data = np.load(os.path.join(PROJECT_ROOT_PATH,
                                                     "visualizing", "statistic", "DE_training_log",
                                                     f"{algo_name}_traininglog_{data_mode}_{data_name}_sample1000_gen{gencount}.npy"))
                axs[idx].plot(training_data[:,0],label=f"history best ")
                axs[idx].plot(training_data[:,1],label=f"avg per gen")
                axs[idx].set_xlabel('gen')  # 添加 x 轴名称
                axs[idx].set_ylabel('util_rate')  # 添加 y 轴名称
                axs[idx].set_title(f'training of {data_mode} {data_name}')
                axs[idx].legend()  # 在每个子图上添加图例
            plt.savefig(f'./pic/training of {algo_name} on {data_mode}')
            plt.show()

def run_test():
    training_data = np.load(r"D:\代码运行数据\traning_log__randomGen_data_Dist_Skyline_1000.npy_atgen100.npy")
    data2 = np.load("traininglog_standard_production_data1_sample1000_gen500.npy")
    plt.plot(training_data[:,0],label="new")
    plt.plot(training_data[:,1])
    # plt.plot(data2[:, 0],label="old")
    # plt.plot(data2[:, 1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()
    pass