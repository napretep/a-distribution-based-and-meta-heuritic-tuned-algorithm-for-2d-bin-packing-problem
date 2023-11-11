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
def run_4data():
    algo_type = [f"{NOISED}Dist2", f"{STANDARD}Dist2", "Skyline", "MaxRect"]

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
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    # 在每个子图上绘制箱线图
    for idx, data in enumerate(data_sets):
        sns.boxplot(x="scale", y="result", hue="algo_name", data=df[df["data_set"] == data], ax=axs[idx])
        axs[idx].set_title(f"Boxplot for {data} with noisedParam_Dist")

    plt.tight_layout()  # 自动调整子图参数，使得子图之间的间距适中
    plt.savefig("4data_compare_on_all_data")
    plt.show()
def run_3data():
    pass

if __name__ == "__main__":
    run_4data()
    pass