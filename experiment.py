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

def main(algo,job_name,parameters=None,data_scales=DATA_SCALES,run_count=RUN_COUNT):
    random_data_result:"list[float]"=[]
    hw_data_result:"list[float]"=[]
    wb_data_result:"list[float]"=[]

    # 每种数据规模,跑30次,对三种数据类型
    for data_scale in data_scales:
        # for i in range(run_count):
        with Pool() as p:
            if parameters:
                algo.scoring_sys.parameters = parameters[0]
            random_data_result+=p.starmap(subprocess_run, [(algo,随机_data,data_scale)]*run_count)
            result = np.array(random_data_result)
            np.save(f"{job_name}_{algo.__class__.__name__}_random_{data_scale}.npy",result)
            if parameters:
                algo.scoring_sys.parameters = parameters[1]
            hw_data_result+=p.starmap(subprocess_run, [(algo,华为杯_data,data_scale)]*run_count)
            result = np.array(random_data_result)
            np.save(f"{job_name}_{algo.__class__.__name__}_hw_{data_scale}.npy", result)
            if parameters:
                algo.scoring_sys.parameters = parameters[2]
            wb_data_result+=p.starmap(subprocess_run, [(algo,外包_data,data_scale)]*run_count)
            result = np.array(random_data_result)
            np.save(f"{job_name}_{algo.__class__.__name__}_wb_{data_scale}.npy", result)





if __name__ == "__main__":

    d = Distribution()
    # d.scoring_sys.parameters =param_hw_100_107
    main(d,job_name="standard",parameters=[param_sj_300_107,param_hw_300_107,param_wb_300_107])
    pass