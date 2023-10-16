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

data_name = "random"
algo_name="Skyline"
# data:"np.ndarray|None"=None
# for scale in DATA_SCALES:
#     temp = np.load(f"../data/结果/{algo_name}_{data_name}_{scale}.npy")
#     print(temp.shape)
#     if data is None:
#         data=temp
#     else:
#         data = np.concatenate((data, temp), axis=0)
# print(data.shape, data)
data = np.load(f"../data/结果/{algo_name}_{data_name}_{5000}.npy")
# 将一维ndarray数据转化为适合Seaborn箱线图的数据格式
data_df = pd.DataFrame({
    'scale': np.repeat(DATA_SCALES, RUN_COUNT),
    'value': data
})
if __name__ == "__main__":

    pass
    # 使用Seaborn的boxplot函数进行可视化
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='scale', y='value', data=data_df)
    plt.title('Algorithm Run Results')
    plt.show()