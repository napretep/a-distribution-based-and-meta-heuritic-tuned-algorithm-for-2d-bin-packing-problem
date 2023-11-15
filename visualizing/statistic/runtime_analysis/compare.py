# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 1:31'
"""
import os

from constant import *
import numpy as np
import BinPacking2DAlgo
from multiprocessing import Pool
# params = {
#         "standard": {
#                 "production_data1": [27.57934813, 23.29018577, 33.43375348, 18.89843672,
#                                      -18.28887118, 0.36416545, 38.55297982, 1.58717868,
#                                      -12.72023321, -4.78548915, 1.24706308, -30.0087219,
#                                      -26.44875766, 19.04054086, -39.76115475, 2.18626198,
#                                      -29.64918256, -14.72861541, 23.58872823, 26.29482364,
#                                      -10.93733512, 2.4618385, 7.3259813, 19.91113574],
#                 "production_data2": [28.12061657, -8.0430912, 22.58183676, -36.278031,
#                                      -18.67314595, 0.14616366, 22.46584206, -35.59192484,
#                                      1.45843571, 2.81054814, -4.70562306, 7.44944437,
#                                      -11.04635553, -25.21184856, 33.64858665, 1.43668068,
#                                      1.38881597, -2.31121838, 37.72415834, 9.3078541,
#                                      8.54222983, 2.6429937, -3.17287881, -9.44685875],
#                 "random_data"     : [25.5435356, 9.25573678, 7.33452141, 11.87821915,
#                                      -18.15776172, -0.69890311, 32.35026717, 20.43240378,
#                                      7.59892088, 5.14509784, -2.94299651, 2.40380363,
#                                      -34.66379054, 8.21110542, -37.56377915, -10.16772923,
#                                      -30.83058312, 26.36755633, 36.43783522, -13.96861355,
#                                      23.04090018, 2.31979539, -7.09244668, -0.84345639],
#         },
#         "random"  : {
#                 "production_data1": [32.5297365, 17.42349883, 31.07130181, -27.08239102,
#                                      -16.1294125, 0.55513815, 24.09125474, 18.33520099,
#                                      -14.46151256, 0.85308145, -0.98344585, -31.19569029,
#                                      -21.8196573, -30.40856389, 0.17618179, -3.6816786,
#                                      28.74556118, 8.17828654, 26.91246714, 5.98856374,
#                                      -17.47834592, -31.059624, -23.15718183, 27.40120483],
#                 "production_data2": [32.81051893, 27.71859658, -18.12926271, 31.47141733,
#                                      11.15234478, 0.98452451, 30.20495797, -13.62208354,
#                                      14.46456117, 0.35245309, 2.57142432, -17.99945398,
#                                      -29.75812519, 24.37060543, -13.10154752, -6.09719204,
#                                      7.50557726, -8.27136646, 36.6475308, 3.24912781,
#                                      -4.3851668, 2.2489736, 35.10086676, 6.805312],
#                 "random_data"     : [20.53944317, -14.50081467, -13.72178025, -36.98244615,
#                                      38.81836636, 0.55734865, 4.25591023, -23.77335997,
#                                      4.96419603, 4.04618501, 5.64926788, -34.76708757,
#                                      -32.87163442, -36.30439092, 8.58333456, 36.94052644,
#                                      9.05199327, -26.73226726, 4.89877997, -39.19794429,
#                                      5.8054671, 12.25104461, 14.58953578, 14.81294095],
#         }
# }

data_sets = {
        PRODUCTION_DATA1: 华为杯_data,
        PRODUCTION_DATA2: 外包_data,
        RANDOMGEN_DATA     : 随机_data,
}
data_types = [STANDARD, NOISED]
scales = [100,300,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
algo_names = [AlgoName.Dist_Skyline, AlgoName.Dist_MaxRect, AlgoName.Skyline, AlgoName.MaxRect]
run_count = 24
def run_experiment():
    with Pool() as p:
        for algo_type in algo_types:
            for data_set in data_sets:
                for scale in scales:
                    start = time()
                    eval_fun = EVAL(algo_type, run_count, params["random"][data_set])
                    input_data = [kde_sample(data_sets[data_set], scale) for _ in range(run_count)]
                    results = p.map(eval_fun.run_time_cost, input_data)
                    path = f"standard_runtime_analysis_{algo_type}_{data_set}_{scale}.npy"
                    np.save(os.path.join(SYNC_PATH,path), np.array(results))
                    print("\n", time() - start,"\n",path)




def run_compare(algo_names):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    side_algos = [algo_names,algo_names[:-1]]
    for col in range(2):
        for algo_type in side_algos[col]:
            mean_datas = []
            for scale in scales:
                data: "np.ndarray|None" = None
                for data_set in data_sets:
                    path = f"standard_runtime_analysis_{algo_type}_{data_set}_{scale}.npy"
                    if data is None:
                        data = np.load(path)
                    else:
                        data = np.row_stack((data, np.load(path)))
                if algo_type == AlgoName.Dist_Skyline:
                    data = data / 1.4
                mean_datas.append(data.mean())
            axs[col].plot(scales, mean_datas, label=f'{algo_type} on all dataset avg')
            for i in range(len(scales)):
                axs[col].annotate(f"{mean_datas[i]:.2f}", (scales[i], mean_datas[i]))

        axs[col].set_xlabel('Scale')
        axs[col].set_ylabel('mean runtime (second)')
        axs[col].set_title('Comparison of Algorithms runtime')
        axs[col].legend()  # Show line names
    fig.savefig(f"{algo_names.__len__()}_algos_runtime_comparison_i711700KF_5ghz_16thread{int(time())}.png")
    plt.show()

if __name__ == "__main__":
    run_compare(algo_names)
    pass