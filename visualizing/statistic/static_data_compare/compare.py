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

algo_types = ["Dist2"]
# scales = [3000, 5000]




def gen_sample_data(data,scale,i):
    return [random_mix(kde_sample(data, scale)[:, 1:], random_ratio=(0,(i+1) / 100)) for _ in range(40)]




class JOB:
    def __init__(self,data_sets,algo_types,param_source,scales=(100,300,500,1000,3000,5000),data_type:"list[str]|str"="standard",algo_prefix="",run_count=36):
        """

        :param data_sets:
        :param algo_types:
        :param param_source:
        :param scales:
        :param data_type: "standard","noised"
        :param algo_prefix:
        :param run_count:
        """
        self.param_source = param_source
        self.data_type = data_type
        self.algo_prefix = algo_prefix
        self.data_sets = data_sets
        self.algo_types=algo_types
        self.scales=scales
        self.run_count=run_count

    def DO(self):
        with Pool() as p:
            for data_set in self.data_sets:
                print(data_set)
                for algo_type in self.algo_types:
                    print(algo_type)
                    eval_obj = EVAL(algo_type, run_count, self.param_source[self.data_type][data_set])
                    for scale in self.scales:
                        timestart = time()
                        if type(self.data_type)==list:
                            for data_type in self.data_type:
                                file_name = f"{data_type}_{self.algo_prefix}{algo_type}_{data_set}_{scale}_.npy"
                                if data_type!="standard":
                                    run_results = []
                                    for i in range(30):
                                        print(file_name, f"random interval=(0,{(i + 1) / 100})")
                                        input_data = [random_mix(kde_sample(self.data_sets[data_set], scale)[:, 1:], random_ratio=(0, (i + 1) / 100)) for _ in range(self.run_count)]
                                        result = p.map(eval_obj.run_single, input_data)
                                        run_results.append(result)
                                        print("\n")
                                    np.save(os.path.join(SYNC_PATH,file_name), np.array(run_results))
                                    print(file_name, "done", time() - timestart)
                                else:
                                    input_data = [kde_sample(self.data_sets[data_set], scale) for _ in range(self.run_count)]
                                    result = p.map(eval_obj.run_single, input_data)
                                    np.save(os.path.join(SYNC_PATH,file_name), np.array(result))
                                    print(file_name, "done", time() - timestart)
                        else:
                            file_name = f"{self.data_type}_{self.algo_prefix}{algo_type}_{data_set}_{scale}_.npy"
                            if self.data_type!="standard":
                                run_results = []
                                for i in range(30):
                                    print(file_name, f"random interval=(0,{(i + 1) / 100})")
                                    input_data = [random_mix(kde_sample(self.data_sets[data_set], scale)[:, 1:], random_ratio=(0, (i + 1) / 100)) for _ in range(self.run_count)]
                                    result = p.map(eval_obj.run_single, input_data)
                                    run_results.append(result)
                                    print("\n")
                                np.save(os.path.join(SYNC_PATH,file_name), np.array(run_results))
                                print(file_name, "done", time() - timestart)
                            else:
                                input_data = [kde_sample(self.data_sets[data_set], scale) for _ in range(self.run_count)]
                                result = p.map(eval_obj.run_single, input_data)
                                np.save(os.path.join(SYNC_PATH,file_name), np.array(result))
                                print(file_name, "done", time() - timestart)

if __name__ == "__main__":
    start_time = time()
    job = JOB(
        data_sets= {
            "production_data1": 华为杯_data,
            "production_data2": 外包_data,
            "randomGen_data"  : 随机_data,
        },
        algo_types=["Skyline","Dist2","MaxRect"],
        param_source=params,
        data_type=["standard","noised"],
        algo_prefix="",
    )
    job.DO()
    print(time()-start_time)
    pass