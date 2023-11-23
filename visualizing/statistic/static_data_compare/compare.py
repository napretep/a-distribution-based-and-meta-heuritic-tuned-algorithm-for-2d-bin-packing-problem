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


def gen_sample_data(data, scale, i):
    return [random_mix(kde_sample(data, scale)[:, 1:], random_ratio=(0, (i + 1) / 100)) for _ in range(40)]


class JOB:
    def __init__(self, data_sets, algo_types, param_source, scales=(100, 300, 500, 1000, 3000, 5000), data_mode: "iter[str]" = (STANDARD,), algo_prefix: "iter"=("",), run_count=36):
        """

        :param data_sets:
        :param algo_types:
        :param param_source:
        :param scales:
        :param data_mode: "standard","noised"
        :param algo_prefix:
        :param run_count:
        """
        self.param_source = param_source
        self.data_mode = data_mode
        self.algo_prefix = algo_prefix
        self.data_sets = data_sets
        self.algo_types:"list[str]" = algo_types
        self.scales = scales
        self.run_count = run_count

    def DO(self):
        with Pool() as p:
            for data_set in self.data_sets:
                for algo_type in self.algo_types:
                    for scale in self.scales:
                        for prefix in (self.algo_prefix if algo_type.startswith("Dist") else [""]):
                            for data_mode in self.data_mode:
                                timestart = time()
                                print(data_mode,data_set,prefix,algo_type,scale)
                                eval_obj = EVAL(algo_type, self.run_count, self.param_source[algo_type][prefix][data_set] if algo_type in self.param_source else None)
                                file_name = f"{data_mode}_{data_set}_{prefix}{algo_type}_{scale}_.npy"
                                if data_mode == NOISED:
                                    self.noised_work(p, file_name, data_set, scale, eval_obj,timestart)
                                    pass
                                else:
                                    self.standard_work(p, file_name, data_set, scale, eval_obj,timestart)
                                    pass

    def noised_work(self,p,file_name,data_set,scale,eval_obj:EVAL,timestart):
        run_results = []
        for i in range(50):
            print(file_name, f"random interval=(0,{(i + 1) / 100})")
            input_data = [random_mix(kde_sample(self.data_sets[data_set], scale)[:, 1:], random_ratio=(0, (i + 1) / 100)) for _ in range(self.run_count)]
            result = p.map(eval_obj.run_single, input_data)
            run_results.append(result)
            print("\n")
        np.save(os.path.join(SYNC_PATH, file_name), np.array(run_results))
        print(file_name, "done", time() - timestart)

    def standard_work(self,p,file_name,data_set,scale,eval_obj:EVAL,timestart):
        input_data = [kde_sample(self.data_sets[data_set], scale) for _ in range(self.run_count)]
        result = p.map(eval_obj.run_single, input_data)
        np.save(os.path.join(SYNC_PATH, file_name), np.array(result))
        print(file_name, "done", time() - timestart)


if __name__ == "__main__":
    start_time = time()
    job = JOB(
            data_sets=data_sets,
            algo_types=[
                    AlgoName.Skyline,AlgoName.Dist_Skyline,
                    AlgoName.MaxRect,AlgoName.Dist_MaxRect
            ],
            param_source=params,
            data_mode=[STANDARD, NOISED],
            algo_prefix=[STANDARD,NOISED],
            scales=(100,300,500,1000,3000,5000),
            run_count=40

    )
    job.DO()
    print(time() - start_time)
    pass
