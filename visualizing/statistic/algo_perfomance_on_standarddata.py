# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = '可视化_查看数据分布.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 4:23'
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have results from three algorithms in three different lists
# Each list contains arrays of results for each data scale, each array contains 40 results
results_alg1 = [np.random.rand(40) for _ in range(6)]
results_alg2 = [np.random.rand(40) for _ in range(6)]
results_alg3 = [np.random.rand(40) for _ in range(6)]

data_scales = np.array([100, 200, 300, 400, 500, 600])

# Prepare data for DataFrame
data = []
for scale, res1, res2, res3 in zip(data_scales, results_alg1, results_alg2, results_alg3):
    for res in res1:
        data.append(['Algorithm1', scale, res])
    for res in res2:
        data.append(['Algorithm2', scale, res])
    for res in res3:
        data.append(['Algorithm3', scale, res])
if __name__ == "__main__":

    pass
# Create DataFrame
    df = pd.DataFrame(data, columns=['Algorithm', 'Data Scale', 'Result'])

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Data Scale', y='Result', hue='Algorithm', data=df)
    plt.title('Comparison of Algorithm Results')
    plt.show()