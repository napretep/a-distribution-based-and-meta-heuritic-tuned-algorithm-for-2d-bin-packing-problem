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
def run():
    algo_type=["single_trained_Dist2","superParamDist2","noiseParam_Dist2","Dist2","Skyline","MaxRect"]

    data_sets=["production_data1","production_data2"]#,"random_data"]
    scales = [100,300,500,1000,3000,5000]
    results = []
    for data in data_sets:
        for algo in algo_type:
            for scale in scales:
                result = np.load(f"./standard_{algo}_{data}_{scale}_.npy")
                for result_i in result:
                    results.append({
                            "algo_name":algo,
                            "data_set":data,
                            "scale":scale,
                            "result":result_i
                    })

    # 将结果转换为 DataFrame
    df = pd.DataFrame(results)

    # 绘制箱线图
    for data in data_sets:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="scale", y="result", hue="algo_name", data=df[df["data_set"] == data])
        plt.title(f"Boxplot for {data} with noisedParam_Dist")
        plt.show()

if __name__ == "__main__":
    run()
    pass