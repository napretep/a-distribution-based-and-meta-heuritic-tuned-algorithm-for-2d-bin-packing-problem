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
全部训练完成时间(秒): 40707.74426817894
[
['production_data1', array([-14.15936449,  -5.51115384, -23.64518177,  -5.15047712,
        -3.33357418,  19.30223264, -34.12354662, -37.72826772,
       -22.94565056,  -9.50349139, -34.3961086 ,  17.95800022,
         8.9030257 ,  15.07774937,  -1.14072358, -26.85316237,
        -4.21159189,  -9.02667882, -18.79295732, -21.38535338,
       -33.5686066 , -16.21340102,  21.69621865,   9.50849564,
        15.90809837,  -3.63610164,  -1.60874255]), 0.93095071464777], 
训练时间(秒): 11454.134886980057
['production_data2', array([-11.96752198,  -2.7668223 ,  38.70687562,  13.18271721,
         3.67444606,  38.25703788, -18.28533271,   3.38803386,
        -6.55048104, -37.50351178, -14.20464962,   9.0491134 ,
        -0.42823428,  11.01235021,  -6.35228745, -35.80967409,
         6.66501912,   2.81858832, -19.88194734,  28.88062709,
       -33.12758532, -25.59401527,  -4.40720866,  -2.71437647,
         8.11387952,  23.83868208,   2.78775707]), 0.9397160932421684], 
训练时间(秒): 16333.756328582764
['random_data', array([ 16.52380377,   8.77870006,  -2.37123285, -18.76190523,
        -9.94336197,  -2.19416358, -24.36524343, -17.02973983,
         1.41957027, -10.40319628,   5.7913216 ,  12.38149365,
        27.32627864,   1.76764851, -17.20481395,  28.93672477,
       -16.06091363,  15.07383287,  -9.77274668, -38.1253096 ,
       -26.23267942, -18.76066081, -36.4631387 ,  -8.67292712,
       -36.63156038,  -0.94854679,  -0.39558535]), 0.9247190743684768]
]
训练时间(秒): 12919.711238145828
"""

def packing_log_vector_to_obj(packinglog:"List[List[List[List[float]]]]"):
    solution = []
    for i in range(len(packinglog)):
        plan_log = packinglog[i]


class DE:
    def __init__(self,data_set,data_set_name,total_param_num=27,eval_run_count=40,data_sample_scale=500,random_ratio=None):
        self.data_set = data_set
        self.data_set_name = data_set_name
        self.total_param_num = total_param_num
        self.eval_run_count = eval_run_count
        self.data_sample_scale = data_sample_scale
        self.time_recorder=[]
        self.training_log=[]
        self.random_ratio = random_ratio
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
        result = BinPacking2DAlgo.multi_run(data_for_run, MATERIAL_SIZE, parameter_input_array=param,run_count=self.eval_run_count)
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
                                        callback=self.callback, maxiter=10)
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