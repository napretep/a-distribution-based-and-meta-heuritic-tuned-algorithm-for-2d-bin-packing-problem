# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/9 21:42'
"""
import numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
import pandas as pd
from constant import *
def run_data_6():
    algo_type = [ AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",f"{NOISED}{AlgoName.Dist_MaxRect}",AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]
    data_sets = [PRODUCTION_DATA1, PRODUCTION_DATA2, RANDOMGEN_DATA]
    scales = [100, 300, 500, 1000, 3000, 5000]
    results = []
    for data in data_sets:
        for algo in algo_type:
            for scale in scales:
                result = np.load(f"./standard_{data}_{algo}_{scale}_.npy")
                for result_i in result:
                    results.append({
                        "algo_name": algo,
                        "data_set": data,
                        "scale": scale,
                        "result": result_i
                    })

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 创建包含三个子图的图像
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))

    # 在每个子图上绘制箱线
    max_rect_df = df[(df["algo_name"] == algo_type[0])|(df["algo_name"] == algo_type[1])|(df["algo_name"] ==algo_type[2])]
    for idx, data in enumerate(data_sets):
        sns.boxplot(x="scale", y="result", hue="algo_name", data=max_rect_df[max_rect_df["data_set"] == data], ax=axs[0,idx])
        axs[0,idx].set_title(f"Boxplot for {data} with noisedParam_Dist")

    skyline_df = df[(df["algo_name"] == algo_type[3]) | (df["algo_name"] == algo_type[4]) | (df["algo_name"] == algo_type[5])]
    for idx, data in enumerate(data_sets):
        sns.boxplot(x="scale", y="result", hue="algo_name", data=skyline_df[skyline_df["data_set"] == data], ax=axs[1,idx])
        axs[1,idx].set_title(f"Boxplot for {data} with noisedParam_Dist")

    plt.tight_layout()  # 自动调整子图参数，使得子图之间的间距适中
    plt.savefig(f"./pic/{6} algos compare on all data_{time()}.png")
    plt.show()
def run_data_4():
    algo_type = [ AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}"]

    data_sets = [PRODUCTION_DATA1, PRODUCTION_DATA2, RANDOMGEN_DATA]
    scales = [100, 300, 500, 1000, 3000, 5000]
    results = []
    for data in data_sets:
        for algo in algo_type:
            for scale in scales:
                result = np.load(f"./standard_{data}_{algo}_{scale}_.npy")
                for result_i in result:
                    results.append({
                        "algo_name": algo,
                        "data_set": data,
                        "scale": scale,
                        "result": result_i
                    })

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 创建包含三个子图的图像
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))

    # 在每个子图上绘制箱线
    max_rect_df = df[(df["algo_name"] == algo_type[0])|(df["algo_name"] == algo_type[1])]
    for idx, data in enumerate(data_sets):
        sns.boxplot(x="scale", y="result", hue="algo_name", data=max_rect_df[max_rect_df["data_set"] == data], ax=axs[0,idx])
        axs[0,idx].set_title(f"Boxplot for {data} with noisedParam_Dist")

    skyline_df = df[(df["algo_name"] == algo_type[2]) | (df["algo_name"] == algo_type[3])]
    for idx, data in enumerate(data_sets):
        sns.boxplot(x="scale", y="result", hue="algo_name", data=skyline_df[skyline_df["data_set"] == data], ax=axs[1,idx])
        axs[1,idx].set_title(f"Boxplot for {data} with noisedParam_Dist")

    plt.tight_layout()  # 自动调整子图参数，使得子图之间的间距适中
    plt.savefig(f"./pic/{4} algos compare on all data_{time()}.png")
    plt.show()


if __name__ == "__main__":
    # run_data(4)
    run_data_6()
    run_data_4()
    pass