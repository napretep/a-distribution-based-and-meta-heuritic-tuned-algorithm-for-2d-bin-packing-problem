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

    params ={
            "production_data1": [ 38.61747086,  -5.95326796,  21.94161095,   8.23033128,  -6.54194618,
  39.71401503,   5.33590528, -29.19421696, -18.68411113,   1.57922967,
 -21.31933665,   2.55612429,  20.24922114,  14.14264984,  -1.72218166,
 -35.46570517, -28.12804252,  -1.79172419,  10.43338687,  -8.44463058,
 -38.96822555,  17.30081879, -25.81221402,  17.44919635, -11.88416998,
  -9.28382087,  -2.42562201,],
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

    data_sets={
            "production_data1":华为杯_data,
            "production_data2":外包_data,
            "random_data":随机_data,
    }
    scales = [100,300,500,1000,3000,5000]
    algo_types=["Dist"]
    run_count=40

    for data_set in data_sets:
        results = []
        print(data_set)
        for algo_type in algo_types:
            print(algo_type)
            for scale in scales:
                print(scale)
                result_data = BinPacking2DAlgo.multi_run([kde_sample(data_sets[data_set],scale) for i in range(run_count)],MATERIAL_SIZE,run_count=run_count,algo_type=algo_type,parameter_input_array=params[data_set])
                np.save(f"standard_{algo_type}_{data_set}_{scale}_{time()}.npy",result_data)
                for i in range(run_count):
                    results.append({"Algorithm": algo_type, "Scale": scale, "Run": i, "Result": result_data[i]})
                print(result_data)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    start_time = time()
    compare_random()
    print(time()-start_time)
    pass