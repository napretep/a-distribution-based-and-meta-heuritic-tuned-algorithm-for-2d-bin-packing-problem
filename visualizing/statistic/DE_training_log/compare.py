# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 5:57'
"""
import itertools
import os

import matplotlib.pyplot as plt

from constant import *


def run():

    training_modes = [STANDARD, NOISED]
    algo_names = [ AlgoName.Dist_MaxRect,AlgoName.Dist_Skyline]
    data_names =list(data_sets.keys())

    for mode in range(2):
        fig, axs = plt.subplots(nrows=2, ncols=len(data_names), figsize=(24, 8))
        for row in range(2):
            for col in range(len(data_names)):
                training_data = np.load(os.path.join(PROJECT_ROOT_PATH,
                                                     "visualizing", "statistic", "DE_training_log",
                                                     f"{algo_names[row]}_traininglog_{training_modes[mode]}_{data_names[col]}_sample1000_gen{500}.npy"))
                axs[row,col].plot(training_data[:,0],label=f"history best ulti_rate")
                axs[row,col].plot(training_data[:,1],label=f"avg ulti_rate per gen")
                axs[row,col].set_xlabel('gen')  # 添加 x 轴名称
                axs[row,col].set_ylabel('util_rate')  # 添加 y 轴名称
                title_name = f'{algo_names[row]},{training_modes[mode]},{data_names[col]}'
                axs[row,col].set_title(title_name)
                axs[row,col].legend()  # 在每个子图上添加图例
        plt.tight_layout()
        plt.savefig(f'./pic/training of algos on {training_modes[mode]}_{int(time())}.png')
        plt.show()


def run_test():
    training_data = np.load(r"D:\代码运行数据\traning_log__randomGen_data_Dist_Skyline_1000.npy_atgen100.npy")
    data2 = np.load("traininglog_standard_production_data1_sample1000_gen500.npy")
    plt.plot(training_data[:,0],label="new")
    plt.plot(training_data[:,1])
    plt.legend()
    plt.show()


def test():
    # training_modes =
    data_names = list(data_sets.keys())
    fig, axs = plt.subplots(nrows=4, ncols=len(data_names), figsize=(24, 8))
    algo_names = list(itertools.product([STANDARD, NOISED], [AlgoName.Dist_MaxRect, AlgoName.Dist_Skyline]))
    for row in range(4):
        mode = row%2
        training_mode,algo_name =algo_names[row]
        for col in range(len(data_names)):
            training_data = np.load(os.path.join(PROJECT_ROOT_PATH,
                                                 "visualizing", "statistic", "DE_training_log",
                                                 f"{algo_name}_traininglog_{training_mode}_{data_names[col]}_sample1000_gen{500}.npy"))


            training_data2 = np.load(os.path.join(PROJECT_ROOT_PATH,
                                                 "visualizing", "statistic", "DE_training_log","old",
                                                 f"{algo_name}_traininglog_{training_mode}_{data_names[col]}_sample1000_gen{500}.npy"))

            print(f"{algo_name}_traininglog_{training_mode}_{data_names[col]}", np.max(training_data[:, 0]-np.max(training_data2[:, 0])))
    #         axs[row, col].plot(training_data[:, 0], label=f"history best ulti_rate")
    #         axs[row, col].plot(training_data[:, 1], label=f"avg ulti_rate per gen")
    #         axs[row, col].set_xlabel('gen')  # 添加 x 轴名称
    #         axs[row, col].set_ylabel('util_rate')  # 添加 y 轴名称
    #         title_name = f'{algo_name},{training_mode},{data_names[col]}'
    #         axs[row, col].set_title(title_name)
    #         axs[row, col].legend()  # 在每个子图上添加图例
    # plt.tight_layout()
    # plt.show()

def get_single_pic():
    # 假定 PROJECT_ROOT_PATH, AlgoName, data_sets 是在上下文中定义好的
    training_modes = [STANDARD, NOISED]
    algo_names = [AlgoName.Dist_MaxRect, AlgoName.Dist_Skyline]
    data_names = list(data_sets.keys())

    # 创建一个目录来保存所有的图
    os.makedirs('./pic', exist_ok=True)

    for mode in training_modes:
        for row, algo in enumerate(algo_names):
            for col, data_name in enumerate(data_names):
                # 加载训练数据
                training_data = np.load(os.path.join(PROJECT_ROOT_PATH,
                                                     "visualizing", "statistic", "DE_training_log",
                                                     f"{algo}_traininglog_{mode}_{data_name}_sample1000_gen{500}.npy"))
                # 创建一个独立的图
                plt.figure(figsize=(8, 8))
                plt.plot(training_data[:, 0], label=f"history best util_rate")
                plt.plot(training_data[:, 1], label=f"avg util_rate per gen")
                plt.xlabel('gen')
                plt.ylabel('util_rate')
                title_name = f'traininglog-{algo}-{mode}-{data_name}'
                plt.title(title_name)
                plt.legend()
                plt.tight_layout()
                # 保存图
                plt.savefig(f'./pic/{title_name}.png')
                plt.close()  # 关闭图形，这样它就不会显示出来了



if __name__ == "__main__":
    get_single_pic()
    pass