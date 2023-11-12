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
    for data_type in [STANDARD, NOISED]:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # 创建一个1行3列的子图
        for idx, data_name in enumerate([PRODUCTION_DATA1, PRODUCTION_DATA2, RANDOMGEN_DATA]):
            training_data = np.load(os.path.join(PROJECT_ROOT_PATH,
                                                 "visualizing", "statistic", "DE_training_log",
                                                 f"traininglog_{data_type}_{data_name}_sample1000_gen500.npy"))
            axs[idx].plot(training_data[:,0],label=f"history best ")
            axs[idx].plot(training_data[:,1],label=f"avg per gen")
            axs[idx].set_xlabel('gen')  # 添加 x 轴名称
            axs[idx].set_ylabel('util_rate')  # 添加 y 轴名称
            axs[idx].set_title(f'training of {data_type} {data_name}')
            axs[idx].legend()  # 在每个子图上添加图例
        plt.savefig(f'training of {data_type}')
        plt.show()
if __name__ == "__main__":
    run()
    pass