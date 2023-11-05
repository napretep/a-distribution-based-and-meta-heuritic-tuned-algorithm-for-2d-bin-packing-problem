# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'result_visual.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/15 15:41'
"""

from constant import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_name = "hw"
# algo_name="MaxRect"
# data:"np.ndarray|None"=None
# for scale in DATA_SCALES:
#     temp = np.load(f"../data/结果/{algo_name}_{data_name}_{scale}.npy")
#     print(temp.shape)
#     if data is None:
#         data=temp
#     else:
#         data = np.concatenate((data, temp), axis=0)
# print(data.shape, data)

# 将一维ndarray数据转化为适合Seaborn箱线图的数据格式
# data_df = pd.DataFrame({
#     'scale': np.repeat(DATA_SCALES[:4], RUN_COUNT),
#     'value': data
# })

def compare_different_algo():


    results_alg1 = np.load(f"../data/结果/Skyline_{data_name}_{5000}.npy").reshape((len(DATA_SCALES),RUN_COUNT))
    results_alg2 = np.load(f"../data/结果/MaxRect_{data_name}_{1000}.npy")
    results_alg2 = np.concatenate((results_alg2,np.ones(len(DATA_SCALES)*RUN_COUNT-results_alg2.shape[0]))).reshape((len(DATA_SCALES),RUN_COUNT))
    results_alg3 = np.load(f"../data/结果/standard_Distribution_{data_name}_{5000}.npy").reshape((len(DATA_SCALES),RUN_COUNT))
    data = []
    for scale, res1, res2, res3 in zip(DATA_SCALES, results_alg1, results_alg2, results_alg3):
        for res in res1:
            data.append(['Skyline', scale, res])
        for res in res2:
            data.append(['MaxRect', scale, res])
        for res in res3:
            data.append(['Distribution', scale, res])
        pass
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Algorithm', 'Data Scale', 'Result'])

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Data Scale', y='Result', hue='Algorithm', data=df)
    plt.title('Comparison of Algorithm Results')
    plt.show()

def algo_compare():
    data = []



if __name__ == "__main__":
    # print(np.load(f"../data/结果/MaxRect_{data_name}_{1000}.npy").reshape(4,40))
    compare_different_algo()
    pass
    # 使用Seaborn的boxplot函数进行可视化
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='scale', y='value', data=data_df)
    # plt.title('Algorithm Run Results')
    # plt.show()