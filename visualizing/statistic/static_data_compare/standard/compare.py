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

def run_data(algo_count= 4):
    if algo_count == 4:
        algo_sets = [ AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}"]
    else:
        algo_sets = [ AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",f"{NOISED}{AlgoName.Dist_MaxRect}",AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]


    scales = [100, 300, 500, 1000, 3000, 5000]
    results = []
    for data_name in data_sets.keys():
        for algo in algo_sets:
            for scale in scales:
                result = np.load(f"./standard_{data_name}_{algo}_{scale}_.npy")
                for result_i in result:
                    results.append({
                        "algo_name": algo,
                        "data_set": data_name,
                        "scale": scale,
                        "result": result_i
                    })

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 创建包含三个子图的图像
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))

    # 在每个子图上绘制箱线
    if algo_count == 4:
        max_rect_df = df[(df["algo_name"] == algo_sets[0])|(df["algo_name"] == algo_sets[1])]
    else:
        max_rect_df = df[(df["algo_name"] == algo_sets[0]) | (df["algo_name"] == algo_sets[1]) | (df["algo_name"] == algo_sets[2])]
    for idx, data_name in enumerate(data_sets):
        data = max_rect_df[max_rect_df["data_set"] == data_name]
        sns.boxplot(x="scale", y="result", hue="algo_name", data=data, ax=axs[0,idx])
        axs[0,idx].set_title(f"{data_name} with MaxRect ")
    if algo_count==4:
        skyline_df = df[(df["algo_name"] == algo_sets[2]) | (df["algo_name"] == algo_sets[3])]
    else:
        skyline_df = df[(df["algo_name"] == algo_sets[3]) | (df["algo_name"] == algo_sets[4]) | (df["algo_name"] == algo_sets[5])]
    for idx, data_name in enumerate(data_sets):
        sns.boxplot(x="scale", y="result", hue="algo_name", data=skyline_df[skyline_df["data_set"] == data_name], ax=axs[1,idx])
        axs[1,idx].set_title(f" {data_name} with Skyline")

    plt.tight_layout()  # 自动调整子图参数，使得子图之间的间距适中
    plt.savefig(f"./pic/{algo_count} algos compare on all data_{time()}.png")
    plt.show()



def 求整体平均():
    original = "original"
    optimized = "optimized"
    algo_sets = {
            "skyline":{
                    original:AlgoName.Skyline,
                    optimized: [f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]
            },
            "maxrect":{
                    original: AlgoName.MaxRect,
                    optimized: [f"{STANDARD}{AlgoName.Dist_MaxRect}", f"{NOISED}{AlgoName.Dist_MaxRect}"]
            }

    }
    algo_families = ["skyline","maxrect"]
    scales = [100, 300, 500, 1000, 3000, 5000]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    # 两组图, 每组图在每个尺度上, 综合所有的分布上的结果
    for col in range(2):
        ax = axs[col]
        # ax2 = ax.twinx()  # Create a second y-axis
        algo_fam = algo_families[col]
        original_algo = algo_sets[algo_fam]["original"]
        standard_op = algo_sets[algo_fam]["optimized"][0]
        noised_op = algo_sets[algo_fam]["optimized"][1]
        mean_data={
        }
        for scale in scales:
            for data_name in data_sets:
                result1 = np.load(f"./standard_{data_name}_{original_algo}_{scale}_.npy")
                result2 = np.load(f"./standard_{data_name}_{standard_op}_{scale}_.npy")
                result3 = np.load(f"./standard_{data_name}_{noised_op}_{scale}_.npy")
                op_result = np.row_stack((result2,result3))
                original_mean = np.mean(result1)
                optimized_mean = np.mean(op_result)
                if scale not in mean_data:
                    mean_data[scale]={
                            original:[],
                            optimized:[]
                    }
                mean_data[scale][original].append(original_mean)
                mean_data[scale][optimized].append(optimized_mean)
        optimized_means = [np.mean(mean_data[scale][optimized]) for scale in mean_data]
        original_means = [np.mean(mean_data[scale][original]) for scale in mean_data]
        # optimized_std = [np.std(mean_data[scale][optimized]) for scale in mean_data]
        # original_std = [np.std(mean_data[scale][original]) for scale in mean_data]
        ax.plot(scales, original_means, label='Original mean')
        ax.plot(scales, optimized_means, label='Optimized mean')
        # ax2.plot(scales, optimized_std, label='Optimized std', color='green')
        # ax2.plot(scales, original_std, label='Original std', color='red')
        # ax2.set_ylabel("std")
        ax.set_xlabel('sample scale')
        ax.set_ylabel('mean ulti rate')
        ax.set_title("avg perfom on "+algo_fam)
        # ax2.legend(loc='upper right')
        ax.legend(loc='upper left')
    plt.savefig(f"./pic/total_avg_compare_{time()}.png")
    plt.show()

def load_data_compare_new():
    algo_type = [AlgoName.Skyline, f"{STANDARD}{AlgoName.Dist_Skyline}"]#[AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}"]
    data_sets = [PRODUCTION_DATA1, PRODUCTION_DATA2, RANDOMGEN_DATA]
    scales = [1000]
    results = []



    for data in data_sets:
        new_data_path = rf"D:\代码运行数据\Dist_MaxRect_traininglog_standard_{data}_sample1000_gen500.npy"
        print(new_data_path)
        print(np.max(np.load(new_data_path)[:,0]))
        for algo in algo_type:

            for scale in scales:
                result = np.load(f"./standard_{data}_{algo}_{scale}_.npy")

                print(f"./standard_{data}_{algo}_{scale}_.npy")
                print(np.mean(result))
                # for result_i in result:
                #     results.append({
                #             "algo_name": algo,
                #             "data_set" : data,
                #             "scale"    : scale,
                #             "result"   : result_i
                #     })
        print("")


def make_single_pic():
    algo_sets = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}", f"{NOISED}{AlgoName.Dist_MaxRect}",
                 AlgoName.Skyline, f"{STANDARD}{AlgoName.Dist_Skyline}", f"{NOISED}{AlgoName.Dist_Skyline}"]

    scales = [100, 300, 500, 1000, 3000, 5000]
    results = []

    for data_name in data_sets.keys():
        for algo in algo_sets:
            for scale in scales:
                result = np.load(f"./standard_{data_name}_{algo}_{scale}_.npy")
                for result_i in result:
                    results.append({
                        "algo_name": algo,
                        "data_set": data_name,
                        "scale": scale,
                        "result": result_i
                    })

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 为每个数据集和算法组合创建一个图表
    for data_name in data_sets.keys():
        for algo_count, algo_group in enumerate([algo_sets[:3], algo_sets[3:]]):
            algo_df = df[df["algo_name"].isin(algo_group) & (df["data_set"] == data_name)]
            plt.figure(figsize=(10, 6))
            sns.boxplot(x="scale", y="result", hue="algo_name", data=algo_df)
            title = f"{data_name} with {'MaxRect' if algo_count == 0 else 'Skyline'} Algorithms"
            plt.title(title)
            plt.tight_layout()
            # 保存每个图表为单独的文件
            plt.savefig(f"./pic/standard_compare_{title}.png")
            plt.close()

if __name__ == "__main__":
    make_single_pic()
    pass