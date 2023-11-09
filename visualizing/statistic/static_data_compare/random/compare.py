# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 4:25'
"""

from constant import *


scales = [100,300,500,1000,3000,5000]

algo_types = ["determDist2","noisedDist2","MaxRect","Skyline"]
# determ param dist algo on random ratio dataset
def run_compare():
    results = []
    for data_set in data_sets:

        for algo in algo_types:
            for scale in scales:
                result_data = np.load(f"./random_ratio(0,30)_{algo}_{data_set}_{500}_.npy")
                for i in range(30):
                    results.append({
                            "algo_name":algo if algo not in ["determDist2","noisedDist2"] else algo[:-1],
                            "data_set":data_set,
                            "scale":scale,
                            "noise_ratio(%)":i+1,
                            "result":result_data[i].mean()
                    })
    df = pd.DataFrame(results)
    for data_set in data_sets:
        df_subset = df[df['data_set'] == data_set]
        plt.figure(figsize=(10, 6))
        for key, grp in df_subset.groupby(['algo_name']):
            grp = grp.sort_values(by='noise_ratio(%)')
            plt.plot(grp['noise_ratio(%)'], grp['result'], label=key)
        plt.xlabel('noise_ratio(%)')
        plt.ylabel('Result Mean')
        plt.title(f'Comparison of Algorithms for {data_set}')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    run_compare()
    pass