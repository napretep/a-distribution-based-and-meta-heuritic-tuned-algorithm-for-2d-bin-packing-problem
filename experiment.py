# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'experiment.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/11 23:23'

  - 三组数据
  - 6种规模: 100, 500, 1000, 5000, 7000, 10000
  - 每种规模跑30次
  - 保存为np文件
  - 用并行算法
"""
from multiprocessing import Pool
import time
from algorithm import *
from constant import *
import numpy as np


def subprocess_run(algo:Algo, base_data_set: "np.ndarray", data_scale:int):
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(base_data_set)
    np.random.seed(int(time.time()*10000) % 4294967296)
    input_data = kde_sample(base_data_set,data_scale)
    algo.load_data(input_data)
    print(algo.task_id,"start",f"scale={data_scale}")
    algo.run()
    print(algo.task_id,"end")
    return algo.avg_util_rate()

def main(algo,data_scales=DATA_SCALES,run_count=RUN_COUNT):
    random_data_result:"list[float]"=[]
    hw_data_result:"list[float]"=[]
    wb_data_result:"list[float]"=[]

    # 每种数据规模,跑30次,对三种数据类型
    for data_scale in data_scales:
        # for i in range(run_count):
        with Pool() as p:
            random_data_result+=p.starmap(subprocess_run, [(algo,随机_data,data_scale)]*run_count)
            result = np.array(random_data_result)
            np.save(f"{algo.__class__.__name__}_random_{data_scale}.npy",result)
            hw_data_result+=p.starmap(subprocess_run, [(algo,华为杯_data,data_scale)]*run_count)
            result = np.array(random_data_result)
            np.save(f"{algo.__class__.__name__}_hw_{data_scale}.npy", result)
            wb_data_result+=p.starmap(subprocess_run, [(algo,外包_data,data_scale)]*run_count)
            result = np.array(random_data_result)
            np.save(f"{algo.__class__.__name__}_wb_{data_scale}.npy", result)





if __name__ == "__main__":

    d = Distribution()
    d.scoring_sys.parameters = [-12.764467729922428, -7.2807524490032804, -17.405272153673526, 11.62060943355495, -17.767676767373285, 13.498788968865574, -3.058679224306764, -17.380930383866435, -17.380008727391687, -19.579085902347263, 15.561194939767207, 2.310615782862815, -5.273339286206582, 1.6631169187587558, -1.906345802422087, -3.3207320056750733, -7.4098035553284936, 12.394940621852495] # 1.07
    main(d)
    pass