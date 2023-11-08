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

"""
全部训练完成时间(秒): 44895.768300294876
[['production_data1', array([-16.41974521,   4.13125489,   9.4103478 ,   8.65741829,
         5.02985403,  19.21117061, -28.81263701, -25.21020407,
       -24.71024175,  -2.48939642,   3.37488031,  35.33964669,
        -5.51130956,   9.95235236,  -0.2135157 , -11.66015814,
       -23.33683873, -25.0762704 ,  18.61611493,   8.3654269 ,
         1.33890311,  10.61668881,  35.66663008,  -4.75955503,
       -26.80569794,  36.15368499, -23.72988799]), 0.9486099237050767, '训练用时(秒):12662.613681316376'], 
       ['production_data2', array([ -8.19959874,   0.74081101,  19.7382237 ,  -6.70081883,
         0.58326179,  36.21222725, -29.69989999, -13.60351553,
       -35.21268337,  -0.07994774,  23.35236385,  12.62602927,
       -18.78039989,  39.88120253,  -5.54283985, -30.85749554,
       -32.96410889, -30.84110472,  22.12749834,  28.93317737,
        -6.11984355,  11.06597756,   2.6359345 ,  16.91978761,
       -17.76328212,  33.70208676,  37.81276862]), 0.959110730733627, '训练用时(秒):15911.02594947815'], 
       ['random_data', array([ 9.99485403e+00, -2.99377785e+00,  3.83139265e+01, -1.97170595e+01,
       -2.58937215e+01,  9.24819159e+00, -3.52827168e+01, -2.10362191e+00,
       -6.95937537e+00, -6.65119904e+00,  1.20648340e+01,  2.52599056e+01,
        1.15373925e+01,  1.35539582e+01, -4.17721234e-03, -4.52321952e+00,
       -2.30055485e+01, -3.77783095e+01,  1.92449964e+01, -1.45313887e+01,
       -1.87820336e+01,  2.64781432e+01,  2.98700118e+00,  2.42949570e+00,
        1.35628839e+01, -2.71968612e+01, -2.37904332e+01]), 0.9212146951602056, '训练用时(秒):16322.128669500351']]

"""

def packing_log_vector_to_obj(packinglog:"List[List[List[List[float]]]]"):
    solution = []
    for i in range(len(packinglog)):
        plan_log = packinglog[i]


class DE:
    def __init__(self,data_set,data_set_name,total_param_num=24 ,eval_run_count=40,data_sample_scale=500,random_ratio=None,algo_name="Dist2",max_iter=500):
        self.data_set = data_set
        self.data_set_name = data_set_name
        self.total_param_num = total_param_num
        self.eval_run_count = eval_run_count
        self.data_sample_scale = data_sample_scale
        self.time_recorder=[]
        self.training_log=[]
        self.random_ratio = random_ratio
        self.algo_name=algo_name
        self.max_iter=max_iter
    def get_sampled_items(self):
        if self.random_ratio is not None:
            return self.random_mix()
        else:
            return kde_sample(self.data_set, self.data_sample_scale)

    def random_mix(self):
        determ_data = kde_sample(self.data_set, self.data_sample_scale)[:,1:]
        random_item_scale = int(np.random.uniform(*self.random_ratio)*self.data_sample_scale)
        determ_data_idx = np.random.choice(self.data_sample_scale,size=self.data_sample_scale-random_item_scale,replace=False)
        determ_data=determ_data[determ_data_idx]
        random_x = (np.random.uniform(0.1,1,random_item_scale)*MATERIAL_SIZE[0]).astype(int)
        random_y = (np.random.uniform(0.1,1,random_item_scale)*MATERIAL_SIZE[1]).astype(int)
        random_data=np.column_stack((random_x,random_y))
        result =  np.row_stack((determ_data,random_data))
        result = np.column_stack((range(self.data_sample_scale),result))
        return result


    def eval(self,param):
        data_for_run = [self.get_sampled_items() for i in range(self.eval_run_count)]
        result = BinPacking2DAlgo.multi_run(data_for_run, MATERIAL_SIZE, parameter_input_array=param,run_count=self.eval_run_count,algo_type=self.algo_name)
        result = np.array(result)
        # print(result)
        mean = np.mean(result)
        std = np.std(result)
        cutoff = std * 3
        lower, upper = mean - cutoff, mean + cutoff
        result = result[(result > lower) & (result < upper)]
        mean = np.mean(result)
        return 1/mean

    def run(self):
        start_time = time()
        self.time_recorder.append(start_time)
        from scipy.optimize import differential_evolution

        bounds = [[-40, 40]] * self.total_param_num

        result = differential_evolution(self.eval, bounds, workers=-1,  strategy="randtobest1exp", popsize=12,tol=0.0001,init="random",mutation=(0.5,1.5),recombination=0.9,
                                        callback=self.callback, maxiter=self.max_iter)
        end_time = time()
        print("训练时间(秒):",end_time - start_time)
        to_save = [end_time, result.x, result.fun]
        print(to_save)
        np.save(f"{self.algo_name}_{self.data_set_name}_{('random_'+self.random_ratio.__str__() )if self.random_ratio is not None else ''}_{self.data_sample_scale}_traning_log__{round(end_time)}.npy", np.array(self.training_log))
        return result.x, result.fun, self.training_log

        pass

    def callback(self, xk, convergence):
        self.time_recorder.append(time())
        self.training_log.append([len(self.time_recorder),1/self.eval(xk)])

        print(f'time cost {self.time_recorder[-1] - self.time_recorder[-2]} Current solution: {list(xk)}, ratio={1/self.eval(xk)} , Convergence: {convergence}')


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
    start_time = time()

    result = []
    for data,name in [[华为杯_data,"production_data1"],[外包_data,"production_data2"],[随机_data,"random_data"]]:
        start_time2=time()
        d = DE(data,name)
        x,fun,log = d.run()
        result.append([name,x,1/fun,f"训练用时(秒):{time()-start_time2}"])
    end_time = time()
    print("全部训练完成时间(秒):", end_time - start_time)
    print(result)