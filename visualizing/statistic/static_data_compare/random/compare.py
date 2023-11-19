# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 4:25'
"""

from constant import *

import  matplotlib.cm as  cm
from scipy.stats import linregress

cmap = cm.get_cmap('Dark2')

scales = [100,300,500,1000,3000,5000]

algo_names = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_MaxRect}",f"{NOISED}{AlgoName.Dist_Skyline}"]
# 定义不同算法类型和比例尺度的颜色和形状
color_map = {f'{STANDARD}Dist2': 'b', f'{NOISED}Dist2': 'g', 'MaxRect': 'r', 'Skyline': 'c'}

colors = cmap(range(len(algo_names)))
# shape_map = {100: 'o', 300: 's', 500: '^', 1000: 'D', 3000: 'x', 5000: '+'}


# determ param dist algo on random ratio dataset
def run_compare_algo(algo_end=6):
    selected_algo_names = algo_names[:algo_end]
    results = []
    for data_set in data_sets:

        for algo in selected_algo_names:
            for scale in scales:
                result_data = np.load(f"./{NOISED}_{data_set}_{algo}_{scale}_.npy")
                for i in range(result_data.shape[0]):
                    for j in range(result_data.shape[1]):
                        results.append({
                                "algo_name":algo ,
                                "data_set":data_set,
                                "scale":scale,
                                "noise_ratio(%)":i+1,
                                "result":result_data[i,j]
                        })
    df = pd.DataFrame(results)

    # 创建一行三列的子图布局
    fig, axs = plt.subplots(1, len(data_sets), figsize=(5* len(data_sets), 5))

    for idx, data_set in enumerate(data_sets):
        ax = axs[idx] if len(data_sets) > 1 else axs
        for algo_id in range(len(selected_algo_names)):
            algo = selected_algo_names[algo_id]

            df_filtered = df[
                (df['data_set'] == data_set) &
                (df['algo_name'] == algo)
                ]
            df_grouped = df_filtered.groupby('noise_ratio(%)')['result'].mean().reset_index()

            # 绘制数据点
            ax.plot(df_grouped['noise_ratio(%)'], df_grouped['result'],
                    color=colors[algo_id],
                    label=f"{algo}")

            # 计算并绘制回归直线
            coeffs = np.polyfit(df_grouped['noise_ratio(%)'], df_grouped['result'], 1)
            poly = np.poly1d(coeffs)
            ax.plot(df_grouped['noise_ratio(%)'], poly(df_grouped['noise_ratio(%)']),
                    color=colors[algo_id], linestyle='dashed')

        ax.set_title(f"different algorithms on {data_set}")
        ax.set_xlabel("Noise Ratio (%)")
        ax.set_ylabel("Result")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"./pic/{algo_end}algo compare on noised data{int(time())}.png")
    plt.show()


if __name__ == "__main__":
    # run_compare_algo(3)
    run_compare_algo()
    pass