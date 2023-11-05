# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'traning.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/19 8:31'
"""
from time import time

import numpy as np

from algorithm import Skyline,Distribution,MaxRect
from algorithm.distribution_based import ScoringSys
from constant import *
from visualizing.draw_plan import standard_draw_plan
import BinPacking2DAlgo

def standard():
    print("start")
    start_time = time()
    np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(华为杯_data.shape[0], 300)
    data = 华为杯_data[data_idx]
    d = Distribution(data)
    best_ind, best_score, log = d.fit_DE()
    print(d.task_id)
    d.scoring_sys.parameters = best_ind
    d.run()
    end_time = time()
    print("time cost=", end_time - start_time)
    print(d.avg_util_rate())
    standard_draw_plan(d.solution, task_id=d.task_id)


def packing_log_vector_to_obj(packinglog:"List[List[List[List[float]]]]"):
    solution = []
    for i in range(len(packinglog)):
        plan_log = packinglog[i]


class DE:
    def __init__(self,data_set,data_set_name,total_param_num=27,eval_run_count=40,data_sample_scale=300):
        self.data_set = data_set
        self.data_set_name = data_set_name
        self.total_param_num = total_param_num
        self.eval_run_count = eval_run_count
        self.data_sample_scale = data_sample_scale
        self.time_recorder=[]
        self.training_log=[]

    def eval(self,param):
        start_time = time()
        data_for_run = [kde_sample(self.data_set, self.data_sample_scale) for i in range(self.eval_run_count)]
        result = BinPacking2DAlgo.multi_run(data_for_run, MATERIAL_SIZE, parameter_input_array=param,run_count=self.eval_run_count)
        result = np.array(result)
        mean = np.mean(result)
        std = np.std(result)
        cutoff = std * 3
        lower, upper = mean - cutoff, mean + cutoff
        result = result[(result > lower) & (result < upper)]
        end_time = time()
        mean = np.mean(result)
        return 1/mean
    def run(self):
        start_time = time()
        self.time_recorder.append(start_time)
        from scipy.optimize import differential_evolution

        bounds = [[-20, 20]] * self.total_param_num
        result = differential_evolution(self.eval, bounds, workers=-1, atol=0.0001, strategy="randtobest1exp", popsize=24,
                                        callback=self.callback, maxiter=1000)
        end_time = time()
        print("训练时间(秒):",end_time - start_time)
        to_save = [end_time, result.x, result.fun]
        print(to_save)
        np.save(f"{self.data_set_name}_{self.data_sample_scale}_traning_log__{round(end_time)}.npy", np.array(self.training_log))
        return result.x, result.fun, self.training_log

        pass

    def callback(self, xk, convergence):
        self.time_recorder.append(time())
        self.training_log.append([len(self.time_recorder),1/self.eval(xk)])

        print(f'time cost {self.time_recorder[-1] - self.time_recorder[-2]} Current solution: {xk}, ratio={1/self.eval(xk)} , Convergence: {convergence}')


def solution_draw(solution:BinPacking2DAlgo.Algo,text=""):
    for i in range(len(solution.packinglog)-1,-1,-1):
        new_solution = []
        plan_log = solution.packinglog[i]
        for j in range(len(plan_log)):
            plan = ProtoPlan(i, Rect(0, 0, *MATERIAL_SIZE), [Item(i, Rect(*item), POS(item[0], item[1])) for item in plan_log[j][0]],
                             remain_containers=[Container(POS(rect[0], rect[1]), POS(rect[2], rect[3])) for rect in plan_log[j][1]])
            new_solution.append(plan)
        standard_draw_plan(new_solution,is_debug=True,task_id=solution.task_id,text=text)

if __name__ == "__main__":
    d = DE(华为杯_data,"华为数据")
    x,fun,log = d.run()
    print(x,fun)
