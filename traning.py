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
from multiprocessing import Pool

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

def packing_log_vector_to_obj(packinglog:"List[List[List[List[float]]]]"):
    solution = []
    for i in range(len(packinglog)):
        plan_log = packinglog[i]


def single_eval(param):
    start = time()
    result:"BinPacking2DAlgo.Algo" = BinPacking2DAlgo.single_run(param[0], MATERIAL_SIZE, parameter_input_array=param[1],
                                         algo_type="Dist2")
    value = result.get_avg_util_rate()
    end = time()
    print(f"{round(end-start,2)}s,{round(value*100,2)}%",end=", ")
class DE:
    def __init__(self,data_set,data_set_name,eval_selector="multi",total_param_num=24,pop_size=20,eval_run_count=40,data_sample_scale=1000,random_ratio=None,algo_name="Dist2",max_iter=500,

                 ):
        """
        :param data_set:
        :param data_set_name:
        :param total_param_num:
        :param eval_run_count:
        :param data_sample_scale:
        :param random_ratio:  (0,0.3)
        :param algo_name:
        :param max_iter:
        """
        self.input_data = None
        self.bounds = [[-2,2]]*total_param_num
        self.mutation = 0.4
        self.crossover = 0.9
        self.p = Pool()
        self.eval_selector = eval_selector # "multi" "single"
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
        self.current_gen = 0
        self.pop_size = pop_size
        self.task_id = "task_"+str(uuid.uuid4())[:8]
        self.log_save_name = f"_traning_log_{self.algo_name}_{self.data_set_name}_{('random_' + self.random_ratio.__str__())if self.random_ratio is not None else ''}_{self.data_sample_scale}_{self.task_id}.npy"
        self.param_save_name =lambda fun: f"_param_{self.algo_name}_{self.data_set_name}_{('random_' + self.random_ratio.__str__())if self.random_ratio is not None else ''}_{self.data_sample_scale}_{round(1/fun,2)}_{self.task_id}.npy"


    def run_v2(self):
        self.time_recorder.append(time())
        # current_best = None

        best_score:"np.ndarray|None" = None
        for best_x,best_fitness,avg_fitness in self.optimizing():
            self.time_recorder.append((time()))
            if best_score is None:
                best_score = np.array([])
            print(f"\ngen={self.current_gen},time_use={round(self.time_recorder[-1]-self.time_recorder[-2],2)}s,score={round(1/best_fitness*100,3)}%,x={best_x}")
            self.training_log.append([1/best_fitness,1/avg_fitness])
            if self.current_gen>0 and self.current_gen%100==0:
                np.save(f"at_gen{self.current_gen}"+self.param_save_name(1/best_fitness),best_x)
                np.save(f"at_gen{self.current_gen}"+self.log_save_name,np.array(self.training_log))

        pass

    def optimizing(self):
        dimensions = self.total_param_num
        pop = np.random.rand(self.pop_size, dimensions)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.mutli_process_single_eval(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        #整体平均和历史最高
        for i in range(self.max_iter):
            self.current_gen=i
            selected_indices = np.random.choice(range(self.pop_size), int(self.pop_size * np.random.uniform(0.7,1)), replace=False)
            for j in selected_indices:
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + np.random.uniform(self.mutation,1) * (b - c), min_b,max_b)
                cross_points = np.random.rand(dimensions) < self.crossover
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                f = self.mutli_process_single_eval(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm

            yield best, fitness[best_idx],np.mean(fitness)

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

    def mutli_process_single_eval(self,param):
        start = time()

        datas = [(self.get_sampled_items(),param) for i in range(self.eval_run_count)]
        result = self.p.map(self.single_eval,datas)
        result = np.array(result)
        # print(result)
        mean = np.mean(result)
        # std = np.std(result)
        # cutoff = std * 3
        # lower, upper = mean - cutoff, mean + cutoff
        # result = result[(result > lower) & (result < upper)]
        # mean = np.mean(result)
        end = time()
        print(f"{round(end - start, 2)}s,{round(mean * 100, 2)}%", end=", ")
        return 1 / mean
    def multi_eval(self,param):
        start = time()
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
        end = time()
        print(f"{round(end-start,2)}s,{round(mean*100,2)}%",end=", ")
        return 1/mean


    @staticmethod
    def single_eval(param):
        start = time()
        result:"BinPacking2DAlgo.Algo" = BinPacking2DAlgo.single_run(param[0], MATERIAL_SIZE, parameter_input_array=param[1],
                                             algo_type="Dist2")
        value = result.get_avg_util_rate()
        end = time()
        # print(f"{round(end-start,2)}s,{round(value*100,2)}%",end=", ")
        return value



    def run(self):
        start_time = time()
        self.time_recorder.append(start_time)
        from scipy.optimize import differential_evolution

        bounds = [[-1, 1]] * self.total_param_num

        result = differential_evolution(self.get_eval(), bounds, workers=-1,  strategy="randtobest1exp", popsize=12,tol=0.0001,init="random",mutation=(0.5,1.5),recombination=0.95,
                                        callback=self.callback, maxiter=self.max_iter)
        end_time = time()
        print("训练时间(秒):",end_time - start_time)
        to_save = [end_time, result.x, result.fun]
        print(to_save)
        np.save(self.log_save_name, np.array(self.training_log))
        np.save(self.param_save_name(result.fun), result.x)
        return result.x, result.fun, self.training_log

        pass

    def callback(self, xk, convergence):
        self.time_recorder.append(time())
        eval_value = self.get_eval()(xk)
        self.training_log.append([len(self.time_recorder),1/eval_value])
        self.current_gen += 1
        if self.current_gen%50==0:
            np.save(f"at_gen_{self.current_gen}_" + self.log_save_name, np.array(self.training_log))
            np.save(f"at_gen_{self.current_gen}_" + self.param_save_name(eval_value), xk)
        print(f'\ncurrent_gen={self.current_gen}, time cost {self.time_recorder[-1] - self.time_recorder[-2]} Current solution: {list(xk)}, ratio={1/eval_value} , Convergence: {convergence}\n')

        self.training_log.append([len(self.time_recorder),1/eval_value])



    def get_eval(self):
        if self.eval_selector=="single":
            return self.single_eval
        else:
            return self.multi_eval

def solution_draw(solution:BinPacking2DAlgo.Algo,text=""):
    for i in range(len(solution.packinglog)-1,-1,-1):
        new_solution = []
        plan_log = solution.packinglog[i]
        for j in range(len(plan_log)):
            plan = ProtoPlan(i, Rect(0, 0, *MATERIAL_SIZE), [Item(i, Rect(*item), POS(item[0], item[1])) for item in plan_log[j][0]],
                             remain_containers=[Container(POS(rect[0], rect[1]), POS(rect[2], rect[3])) for rect in plan_log[j][1]])
            new_solution.append(plan)
        standard_draw_plan(new_solution,is_debug=True,task_id=solution.task_id,text=text)




class Training:

    def __init__(self,data_set,training_type="determ",):
        """

        :param training_type: "determ","noised"
        """
        self.training_type = training_type
        self.data_set=data_set

    def run(self):
        start_time = time()

        result = []
        for data, name in self.data_set:
            start_time2 = time()
            d = DE(data, name, random_ratio=(0, 0.3) if self.training_type=="noised" else None, max_iter=500, data_sample_scale=1500)
            # x, fun, log = d.run()
            d.run_v2()
            # result.append([name, x, 1 / fun, f"训练用时(秒):{time() - start_time2}"])
            # np.save(f"{self.training_type}_Dist_{name}_{fun}__{round(time())}.npy", np.array(x))
        end_time = time()
        print("全部训练完成时间(秒):", end_time - start_time)

if __name__ == "__main__":
    t = Training([
            [华为杯_data, "production_data1"],
            [外包_data, "production_data2"],
            [随机_data, "random_data"]
    ],
            training_type="noised"
    )
    t.run()