# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/6 22:34'
"""
import numpy as np

from constant import *
import BinPacking2DAlgo
import pandas as pd
import seaborn as sns
import matplotlib as plt
from multiprocessing import Pool

params = {
        "standard": {
                "production_data1": [27.57934813, 23.29018577, 33.43375348, 18.89843672,
                                     -18.28887118, 0.36416545, 38.55297982, 1.58717868,
                                     -12.72023321, -4.78548915, 1.24706308, -30.0087219,
                                     -26.44875766, 19.04054086, -39.76115475, 2.18626198,
                                     -29.64918256, -14.72861541, 23.58872823, 26.29482364,
                                     -10.93733512, 2.4618385, 7.3259813, 19.91113574],
                "production_data2": [28.12061657, -8.0430912, 22.58183676, -36.278031,
                                     -18.67314595, 0.14616366, 22.46584206, -35.59192484,
                                     1.45843571, 2.81054814, -4.70562306, 7.44944437,
                                     -11.04635553, -25.21184856, 33.64858665, 1.43668068,
                                     1.38881597, -2.31121838, 37.72415834, 9.3078541,
                                     8.54222983, 2.6429937, -3.17287881, -9.44685875],
                "random_data"     : [25.5435356, 9.25573678, 7.33452141, 11.87821915,
                                     -18.15776172, -0.69890311, 32.35026717, 20.43240378,
                                     7.59892088, 5.14509784, -2.94299651, 2.40380363,
                                     -34.66379054, 8.21110542, -37.56377915, -10.16772923,
                                     -30.83058312, 26.36755633, 36.43783522, -13.96861355,
                                     23.04090018, 2.31979539, -7.09244668, -0.84345639],
        },
        "random"  : {
                "production_data1": [32.5297365, 17.42349883, 31.07130181, -27.08239102,
                                     -16.1294125, 0.55513815, 24.09125474, 18.33520099,
                                     -14.46151256, 0.85308145, -0.98344585, -31.19569029,
                                     -21.8196573, -30.40856389, 0.17618179, -3.6816786,
                                     28.74556118, 8.17828654, 26.91246714, 5.98856374,
                                     -17.47834592, -31.059624, -23.15718183, 27.40120483],
                "production_data2": [32.81051893, 27.71859658, -18.12926271, 31.47141733,
                                     11.15234478, 0.98452451, 30.20495797, -13.62208354,
                                     14.46456117, 0.35245309, 2.57142432, -17.99945398,
                                     -29.75812519, 24.37060543, -13.10154752, -6.09719204,
                                     7.50557726, -8.27136646, 36.6475308, 3.24912781,
                                     -4.3851668, 2.2489736, 35.10086676, 6.805312],
                "random_data"     : [20.53944317, -14.50081467, -13.72178025, -36.98244615,
                                     38.81836636, 0.55734865, 4.25591023, -23.77335997,
                                     4.96419603, 4.04618501, 5.64926788, -34.76708757,
                                     -32.87163442, -36.30439092, 8.58333456, 36.94052644,
                                     9.05199327, -26.73226726, 4.89877997, -39.19794429,
                                     5.8054671, 12.25104461, 14.58953578, 14.81294095],
        }
}

data_sets = {
        "production_data1": 华为杯_data,
        "production_data2": 外包_data,
        "random_data"     : 随机_data,
}
data_types = ["standard", "random"]
scales = [3000, 5000]
algo_types = ["Dist2"]
run_count = 36
def compare_random():
    params ={
            "production_data1": [-14.15936449,  -5.51115384, -23.64518177,  -5.15047712,
        -3.33357418,  19.30223264, -34.12354662, -37.72826772,
       -22.94565056,  -9.50349139, -34.3961086 ,  17.95800022,
         8.9030257 ,  15.07774937,  -1.14072358, -26.85316237,
        -4.21159189,  -9.02667882, -18.79295732, -21.38535338,
       -33.5686066 , -16.21340102,  21.69621865,   9.50849564,
        15.90809837,  -3.63610164,  -1.60874255],
            "production_data2": [-11.96752198,  -2.7668223 ,  38.70687562,  13.18271721,
         3.67444606,  38.25703788, -18.28533271,   3.38803386,
        -6.55048104, -37.50351178, -14.20464962,   9.0491134 ,
        -0.42823428,  11.01235021,  -6.35228745, -35.80967409,
         6.66501912,   2.81858832, -19.88194734,  28.88062709,
       -33.12758532, -25.59401527,  -4.40720866,  -2.71437647,
         8.11387952,  23.83868208,   2.78775707],
            "random_data"     : [ 16.52380377,   8.77870006,  -2.37123285, -18.76190523,
        -9.94336197,  -2.19416358, -24.36524343, -17.02973983,
         1.41957027, -10.40319628,   5.7913216 ,  12.38149365,
        27.32627864,   1.76764851, -17.20481395,  28.93672477,
       -16.06091363,  15.07383287,  -9.77274668, -38.1253096 ,
       -26.23267942, -18.76066081, -36.4631387 ,  -8.67292712,
       -36.63156038,  -0.94854679,  -0.39558535],
    }
    data_sets = {
            "production_data1": 华为杯_data,
            "production_data2": 外包_data,
            "random_data"     : 随机_data,
    }
    scales = [100, 300, 500, 1000, 3000, 5000]
    algo_types = ["Skyline"]
    run_count=40
    results=None
    for data_set in data_sets:
        results=[]
        # algotype[scale[randomratio1[result1,...],randomratio2,...]
        for algo_type in algo_types:
            for scale in scales:
                random_ratio_result=[]
                for i in range(1,31):
                    gen_datas = [random_mix(kde_sample(data_sets[data_set],scale)[:,1:],random_ratio=(0,i/100)) for _ in range(run_count)]
                    result_data = BinPacking2DAlgo.multi_run(gen_datas,MATERIAL_SIZE,run_count=run_count,algo_type=algo_type,parameter_input_array=params[data_set])
                    random_ratio_result.append(result_data)
                    print(result_data)
                np.save(f"random_ratio(1,30)_{algo_type}_{data_set}_{scale}_{time()}.npy",np.array(random_ratio_result))


def compare():
    df = pd.DataFrame(columns=["Algorithm", "Scale", "Run", "Result"])

    """
    确定性
    全部训练完成时间(秒): 57250.41469120979 500次
    [['production_data1', array([ 27.57934813,  23.29018577,  33.43375348,  18.89843672,
           -18.28887118,   0.36416545,  38.55297982,   1.58717868,
           -12.72023321,  -4.78548915,   1.24706308, -30.0087219 ,
           -26.44875766,  19.04054086, -39.76115475,   2.18626198,
           -29.64918256, -14.72861541,  23.58872823,  26.29482364,
           -10.93733512,   2.4618385 ,   7.3259813 ,  19.91113574]), 0.9599604740738868, '训练用时(秒):11248.336431026459'], 
    ['production_data2', array([ 28.12061657,  -8.0430912 ,  22.58183676, -36.278031  ,
           -18.67314595,   0.14616366,  22.46584206, -35.59192484,
             1.45843571,   2.81054814,  -4.70562306,   7.44944437,
           -11.04635553, -25.21184856,  33.64858665,   1.43668068,
             1.38881597,  -2.31121838,  37.72415834,   9.3078541 ,
             8.54222983,   2.6429937 ,  -3.17287881,  -9.44685875]), 0.9649471915685213, '训练用时(秒):12825.52998805046'], 
    ['random_data', array([ 25.5435356 ,   9.25573678,   7.33452141,  11.87821915,
           -18.15776172,  -0.69890311,  32.35026717,  20.43240378,
             7.59892088,   5.14509784,  -2.94299651,   2.40380363,
           -34.66379054,   8.21110542, -37.56377915, -10.16772923,
           -30.83058312,  26.36755633,  36.43783522, -13.96861355,
            23.04090018,   2.31979539,  -7.09244668,  -0.84345639]), 0.9345276623964309, '训练用时(秒):33176.54827213287']]
    随机性
    全部训练完成时间(秒): 30622.85742497444 250次
    [['production_data1', array([ 32.5297365 ,  17.42349883,  31.07130181, -27.08239102,
           -16.1294125 ,   0.55513815,  24.09125474,  18.33520099,
           -14.46151256,   0.85308145,  -0.98344585, -31.19569029,
           -21.8196573 , -30.40856389,   0.17618179,  -3.6816786 ,
            28.74556118,   8.17828654,  26.91246714,   5.98856374,
           -17.47834592, -31.059624  , -23.15718183,  27.40120483]), 0.9553601099894597, '训练用时(秒):6811.973150014877'], 
    ['production_data2', array([ 32.81051893,  27.71859658, -18.12926271,  31.47141733,
            11.15234478,   0.98452451,  30.20495797, -13.62208354,
            14.46456117,   0.35245309,   2.57142432, -17.99945398,
           -29.75812519,  24.37060543, -13.10154752,  -6.09719204,
             7.50557726,  -8.27136646,  36.6475308 ,   3.24912781,
            -4.3851668 ,   2.2489736 ,  35.10086676,   6.805312  ]), 0.9656827893012611, '训练用时(秒):7960.055242538452'], 
    ['random_data', array([ 20.53944317, -14.50081467, -13.72178025, -36.98244615,
            38.81836636,   0.55734865,   4.25591023, -23.77335997,
             4.96419603,   4.04618501,   5.64926788, -34.76708757,
           -32.87163442, -36.30439092,   8.58333456,  36.94052644,
             9.05199327, -26.73226726,   4.89877997, -39.19794429,
             5.8054671 ,  12.25104461,  14.58953578,  14.81294095]), 0.9462488553462883, '训练用时(秒):15850.829032421112']]

    """

    with Pool() as p:

        for data_set in data_sets:
            print(data_set)
            for algo_type in algo_types:
                print(algo_type)

                for data_type in data_types:
                    if data_type == "standard":
                        eval_obj = EVAL(algo_type, run_count, params[data_type][data_set])
                        input_data = [[kde_sample(data_sets[data_set],scale) for i in range(run_count)] for scale in scales]
                        result = p.map(eval_obj.run,input_data)
                        for i in range(len(scales)):
                            file_name = f"{data_type}_{algo_type}_{data_set}_{scales[i]}_{time()}.npy"
                            np.save(file_name,np.array(result[i]))
                            print(file_name,"done")
                    else:
                        eval_obj = EVAL(algo_type, run_count, params[data_type][data_set])
                        for scale in scales:
                            input_data = [[gen_sample_data(data_sets[data_set],scale,i) for _ in range(run_count)] for i in range(30)]
                            result = p.map(eval_obj.run, input_data)
                            file_name = f"{data_type}_ratio(0,30)_{algo_type}_{data_set}_{scale}_{time()}.npy"
                            np.save(file_name, np.array(result))
                            print(file_name, "done")

    # 显示图形
    plt.show()

def gen_sample_data(data,scale,i):
    return [random_mix(kde_sample(data, scale)[:, 1:], random_ratio=(0,(i+1) / 100)) for _ in range(40)]



def start_singlerun_compare_job():
    with Pool() as p:
        for data_set in data_sets:
            print(data_set)
            for algo_type in algo_types:
                print(algo_type)
                eval_obj = EVAL(algo_type, run_count, params["random"][data_set])
                for scale in scales:
                    results=[]
                    file_name = f"random_ratio(0,30)_{algo_type}_{data_set}_{scale}_.npy"
                    for i in range(30):
                        print(file_name,f"random interval=(0,{(i+1)/100})")
                        input_data = [random_mix(kde_sample(data_sets[data_set], scale)[:, 1:], random_ratio=(0,(i+1) / 100)) for _ in range(run_count)]
                        result = p.map(eval_obj.run_single, input_data)
                        results.append(result)
                    np.save(file_name, np.array(result))
                    print(file_name, "done")




def compare_noise_param():
    algo_type = "Dist2"
    with Pool() as p:
        for data_set in data_sets:
            eval_obj = EVAL(algo_type, run_count, params["random"][data_set])
            input_data = [[kde_sample(data_sets[data_set], scale) for i in range(run_count)] for scale in scales]
            result = p.map(eval_obj.run, input_data)
            for i in range(len(scales)):
                file_name = f"standard_noiseParam_{algo_type}_{data_set}_{scales[i]}_{time()}.npy"
                np.save(file_name, np.array(result[i]))
                print(file_name, "done")

if __name__ == "__main__":
    start_time = time()
    start_singlerun_compare_job()
    print(time()-start_time)
    pass