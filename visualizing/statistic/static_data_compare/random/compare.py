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
# scales = [1000,3000,5000]
# algo_names = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",f"{NOISED}{AlgoName.Dist_MaxRect}"]#,AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]
algo_names = [AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]
# 定义不同算法类型和比例尺度的颜色和形状
color_map = {f'{STANDARD}Dist2': 'b', f'{NOISED}Dist2': 'g', 'MaxRect': 'r', 'Skyline': 'c'}

colors = cmap(range(len(algo_names)))

def run_compare_algo(algo_count=3):
    if algo_count==2:
        skyline_algo_names = [AlgoName.Skyline, f"{STANDARD}{AlgoName.Dist_Skyline}",]
        maxrect_algo_names = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",]
    else:
        skyline_algo_names = [AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]
        maxrect_algo_names = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",f"{NOISED}{AlgoName.Dist_MaxRect}"]
    selected_algo_sets = [maxrect_algo_names,skyline_algo_names]
    results = []

    for data_set in data_sets:
        for algo_sets in selected_algo_sets:
            for algo_name in algo_sets:
                for scale in scales:
                    result_data = np.load(f"./{NOISED}_{data_set}_{algo_name}_{scale}_.npy")

                    # Apply 3-sigma rule to remove outliers
                    for idx in range(result_data.shape[0]):
                        row_data = result_data[idx, :]
                        mean = np.mean(row_data)
                        std = np.std(row_data)
                        cutoff = std * 3
                        lower, upper = mean - cutoff, mean + cutoff
                        valid_data = (row_data > lower) & (row_data < upper)
                        result_data[idx, :] = row_data * valid_data + mean * ~valid_data

                    for i in range(result_data.shape[0]):
                        for j in range(result_data.shape[1]):
                            results.append({
                                    "algo_name"     : algo_name,
                                    "data_set"      : data_set,
                                    "scale"         : scale,
                                    "noise_ratio(%)": i + 1,
                                    "result"        : result_data[i, j]
                            })

    df = pd.DataFrame(results)

    # Create subplots with 1 row and as many columns as there are data sets
    fig, axs = plt.subplots(2, len(data_sets), figsize=(5 * len(data_sets), 5))

    for row in range(2):
        for col, data_set in enumerate(data_sets):
            ax=axs[row,col]
            for algo_id in range(len(selected_algo_sets[row])):
                algo_name = selected_algo_sets[row][algo_id]

                df_filtered = df[
                    (df['data_set'] == data_set) &
                    (df['algo_name'] == algo_name)
                    ]
                df_grouped = df_filtered.groupby('noise_ratio(%)')['result'].mean().reset_index()

                # Plot the data points
                ax.plot(df_grouped['noise_ratio(%)'], df_grouped['result'],
                        color=colors[algo_id],
                        label=f"{algo_name}")

                # Calculate and plot the regression line
                coeffs = np.polyfit(df_grouped['noise_ratio(%)'], df_grouped['result'], 1)
                poly = np.poly1d(coeffs)
                ax.plot(df_grouped['noise_ratio(%)'], poly(df_grouped['noise_ratio(%)']),
                        color=colors[algo_id], linestyle='dashed')

            ax.set_title(f"different algorithms on {data_set}")
            ax.set_xlabel("Noise Ratio (%)")
            ax.set_ylabel("Result")
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"./pic/{algo_count}algo compare on noised data{int(time())}.png")
    plt.show()



def run_compare_algo2(algo_count=3):
    if algo_count==2:
        skyline_algo_names = [AlgoName.Skyline, f"{STANDARD}{AlgoName.Dist_Skyline}",]
        maxrect_algo_names = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",]
    else:
        skyline_algo_names = [AlgoName.Skyline,f"{STANDARD}{AlgoName.Dist_Skyline}",f"{NOISED}{AlgoName.Dist_Skyline}"]
        maxrect_algo_names = [AlgoName.MaxRect, f"{STANDARD}{AlgoName.Dist_MaxRect}",f"{NOISED}{AlgoName.Dist_MaxRect}"]
    algo_families = [maxrect_algo_names,skyline_algo_names]
    algo_family_name = [AlgoName.MaxRect,AlgoName.Skyline]
    results = []

    for data_set in data_sets:
        for algo_family in algo_families:
            for algo_name in algo_family:
                for scale in scales:
                    result_data = np.load(f"./{NOISED}_{data_set}_{algo_name}_{scale}_.npy")

                    # Apply 3-sigma rule to remove outliers
                    for idx in range(result_data.shape[0]):
                        row_data = result_data[idx, :]
                        mean = np.mean(row_data)
                        std = np.std(row_data)
                        cutoff = std * 3
                        lower, upper = mean - cutoff, mean + cutoff
                        valid_data = (row_data > lower) & (row_data < upper)
                        result_data[idx, :] = row_data * valid_data + mean * ~valid_data

                    for i in range(result_data.shape[0]):
                        for j in range(result_data.shape[1]):
                            results.append({
                                    "algo_name"     : algo_name,
                                    "data_set"      : data_set,
                                    "scale"         : scale,
                                    "noise_ratio(%)": i + 1,
                                    "result"        : result_data[i, j]
                            })

    df = pd.DataFrame(results)
    grouped_df = df.groupby(["noise_ratio(%)", "algo_name"])["result"].mean().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    for col, algo_family in enumerate(algo_families):
        ax = axs[col]
        ax.set_title(f"Algorithm {algo_family_name[col]}")
        ax.set_xlabel("noise_ratio(%)")
        ax.set_ylabel("result")

        for algo_name in algo_family:
            family_data = grouped_df[grouped_df["algo_name"] == algo_name]
            ax.plot(family_data["noise_ratio(%)"], family_data["result"], label=algo_name)

        ax.legend()

    plt.tight_layout()
    plt.savefig(f"./pic/total_count_{time()}.png")
    plt.show()


if __name__ == "__main__":
    # run_compare_algo(3)
    run_compare_algo2(3)
    pass