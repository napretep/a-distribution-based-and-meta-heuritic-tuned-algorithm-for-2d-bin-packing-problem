# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'traning.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/19 8:31'
"""

from constant import *
from visualizing.draw_plan import standard_draw_plan
import BinPacking2DAlgo
import multiprocessing
import gc

class EvalSelect:
    Multi="Multi"
    Single="Single"


@dataclasses.dataclass
class SingleAlgoArgs:
    weights: np.ndarray
    algo_name: str
    data_sample_scale: int
    data_set_name: str
    random_ratio: tuple | None

@dataclasses.dataclass
class DE_MultiArgs:
    data_set_name:str
    idvl_idx:int
    pop:np.ndarray
    pop_size:int
    mutation:float
    crossover:float
    dimensions:int
    diff:np.ndarray
    min_b:np.ndarray
    max_b:np.ndarray
    random_ratio:tuple|None
    data_sample_scale:int
    eval_run_count:int
    algo_name:str
    pop_fitness:np.ndarray

    def single_run(self):
        pass

    def multi_run(self):
        pass

def packing_log_vector_to_obj(packinglog: "List[List[List[List[float]]]]"):
    solution = []
    for i in range(len(packinglog)):
        plan_log = packinglog[i]
def get_stack_vars()->"list[dict]":
    stack=[]
    for frame_info in inspect.stack():
        stack.append(frame_info.frame.f_locals)
    return stack



def sub_process_idvl_eval(arg: SingleAlgoArgs):

    input_data = kde_sample(data_sets[arg.data_set_name],arg.data_sample_scale)
    if arg.random_ratio is not None:
        input_data = random_mix(input_data[:,1:],arg.random_ratio)
    result: "BinPacking2DAlgo.Algo" = BinPacking2DAlgo.single_run(input_data, MATERIAL_SIZE,
                                                                  parameter_input_array=arg.weights,
                                                                  algo_type=arg.algo_name,)
    value = result.get_avg_util_rate()

    return value

def sub_process_DE_pop_eval(arg:DE_MultiArgs):
    # print(arg)
    idxs = [idx for idx in range(arg.pop_size) if idx != arg.idvl_idx]
    a, b, c = arg.pop[np.random.choice(idxs, 3, replace=False)]
    mutant = a + np.random.uniform(arg.mutation, 1) * (b - c)
    mutant = np.where(mutant < 0, 0, mutant)
    mutant = np.where(mutant > 1, 1, mutant)
    cross_points = np.random.rand(arg.dimensions) < arg.crossover
    if not np.any(cross_points):
        cross_points[np.random.randint(0, arg.dimensions)] = True
    trial = np.where(cross_points, mutant, arg.pop[arg.idvl_idx])
    trial = np.round(trial, 4)
    trial_denorm = arg.min_b + trial * arg.diff
    trial_denorm = np.where(trial_denorm < arg.min_b, arg.min_b, trial_denorm)
    trial_denorm = np.where(mutant > arg.max_b, arg.max_b, trial_denorm)
    data_set = data_sets[arg.data_set_name]
    input_data = [ kde_sample(data_set,arg.data_sample_scale)
                   if arg.random_ratio is None
                   else random_mix(kde_sample(data_set,arg.data_sample_scale)[:,1:],arg.random_ratio)
                   for k in range(arg.eval_run_count)]
    results = BinPacking2DAlgo.multi_run(input_data,MATERIAL_SIZE,parameter_input_array=trial_denorm,algo_type=arg.algo_name,run_count=arg.eval_run_count)
    fitness = np.mean(1/np.array(results))
    for obj in list(locals().keys())[:]:
        if not (locals()[obj]  is  fitness or locals()[obj]  is  trial_denorm or locals()[obj]  is  trial or locals()[obj]  is  arg.idvl_idx):
            del locals()[obj]
    gc.collect()
    return fitness,trial_denorm,trial,arg.idvl_idx



class scores_evaluator:
    def __init__(self,p:multiprocessing.Pool,data_name,data_sample_scale,random_ratio,eval_run_count,algo_name):
        self.data_name = data_name
        self.data_sample_scale = data_sample_scale
        self.random_ratio=random_ratio
        self.eval_run_count=eval_run_count
        self.algo_name = algo_name
        self.p=p
        pass

    def run_pop(self, trial_denorm):
        data_set = data_sets[self.data_name]
        input_data = [kde_sample(data_set, self.data_sample_scale)
                      if self.random_ratio is None
                      else random_mix(kde_sample(data_set, self.data_sample_scale)[:, 1:], self.random_ratio)
                      for k in range(self.eval_run_count)]
        result = BinPacking2DAlgo.multi_run(input_data, MATERIAL_SIZE, parameter_input_array=trial_denorm, algo_type=self.algo_name, run_count=self.eval_run_count)
        result = 1 / np.array(result)
        mean = np.mean(result)
        std = np.std(result)
        cutoff = std * 3
        lower, upper = mean - cutoff, mean + cutoff
        result = result[(result > lower) & (result < upper)]
        mean = np.mean(result)

        return mean

    def run_idvl(self,weights):
        start = time()

        datas = [SingleAlgoArgs(weights=weights,
                                algo_name=self.algo_name,
                                data_sample_scale=self.data_sample_scale,
                                data_set_name=self.data_name,
                                random_ratio=self.random_ratio)
                 for i in range(self.eval_run_count)]
        result = self.p.map(sub_process_idvl_eval, datas)
        result = np.array(result)
        # print(result)
        mean = np.mean(result)
        std = np.std(result)
        cutoff = std * 3
        lower, upper = mean - cutoff, mean + cutoff
        result = result[(result > lower) & (result < upper)]
        mean = np.mean(result)
        end = time()
        print(f"{round(end - start, 2)}s,{round(mean * 100, 2)}%", end=", ")
        return 1/mean

class Optimizer:
    def __init__(self,p:multiprocessing.Pool, data_set, data_set_name, eval_selector=EvalSelect.Single, pop_size=20, eval_run_count=40,
                 data_sample_scale=1000, random_ratio=None, algo_name=AlgoName.Dist_MaxRect, max_iter=500,
                 ):
        """
        :param data_set:
        :param data_set_name:
        :param eval_run_count:
        :param data_sample_scale:
        :param random_ratio:  (0,0.3)
        :param algo_name:
        :param max_iter:
        """
        self.total_param_num = BinPacking2DAlgo.get_algo_parameters_length(algo_name)
        self.input_data = None
        self.bounds = [[0, 5]] * self.total_param_num
        self.mutation = 0.2
        self.crossover = 0.9
        self.p = p
        self.eval_selector = eval_selector  # "multi" "single"
        self.data_set = data_set
        self.data_set_name = data_set_name
        self.eval_run_count = eval_run_count
        self.data_sample_scale = data_sample_scale
        self.time_recorder = []
        self.training_log = []
        self.random_ratio = random_ratio
        self.algo_name:str = algo_name
        self.max_iter = max_iter
        self.current_gen = 0
        self.pop_size = pop_size
        self.task_id = "task_" + str(uuid.uuid4())[:8]
        self.log_save_name = f"training_log_{(NOISED + '_') if self.random_ratio is not None else ''}_{self.data_set_name}_{self.algo_name}_{self.data_sample_scale}.npy"
        self.param_save_name = lambda \
            fun: f"{(NOISED + '_') if self.random_ratio is not None else ''}_{self.data_set_name}_param_{self.algo_name}_{self.data_sample_scale}_{round(fun, 2)}.npy"

    def run_v2(self,optimizer):
        self.time_recorder.append(time())
        # current_best = None

        best_score: "np.ndarray|None" = None
        final_best_x: "np.ndarray|None" = None
        for best_x, best_fitness, pop, fitness in optimizer():
            std_value = np.std(np.std(pop, axis=0))
            avg_fitness = np.mean(fitness)
            self.time_recorder.append((time()))
            if best_score is None:
                best_score = np.array([])
            print(
                f"\ngen={self.current_gen},time_use={round(self.time_recorder[-1] - self.time_recorder[-2], 2)}s,pop_std={std_value},avg_score={round(1 / avg_fitness * 100, 3)}%,hist_best_score={round(1 / best_fitness * 100, 3)}%,x={list(best_x)}")
            self.training_log.append([1 / best_fitness, 1 / avg_fitness])
            if (self.current_gen + 1) % 100 == 0:
                np.save(
                    os.path.join(SYNC_PATH, f"{self.algo_name}_param_{NOISED if self.random_ratio is not None else STANDARD}_{self.data_set_name}_{self.data_sample_scale}_gen{self.max_iter}_atgen{self.current_gen + 1}"),
                    best_x)
                np.save(os.path.join(SYNC_PATH, f"{self.algo_name}_traininglog_{NOISED if self.random_ratio is not None else STANDARD}_{self.data_set_name}_{self.data_sample_scale}_gen{self.max_iter}_atgen{self.current_gen + 1}"),
                        np.array(self.training_log))
            final_best_x=best_x
        np.save(os.path.join(SYNC_PATH,
                             f"{self.algo_name}_param_{NOISED if self.random_ratio is not None else STANDARD}_{self.data_set_name}_sample{self.data_sample_scale}_gen{self.max_iter}"), final_best_x)
        np.save(os.path.join(SYNC_PATH,
                             f"{self.algo_name}_traininglog_{NOISED if self.random_ratio is not None else STANDARD}_{self.data_set_name}_sample{self.data_sample_scale}_gen{self.max_iter}"), np.array(self.training_log))

        pass

    def DE(self):
        se = scores_evaluator(self.p, self.data_set_name, self.data_sample_scale, self.random_ratio, self.eval_run_count, self.algo_name)
        print("optimizer init")
        dimensions = self.total_param_num
        pop = np.round(np.random.rand(self.pop_size, dimensions),4)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([se.run_idvl(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best_x = pop_denorm[best_idx]
        best_f = fitness[best_idx]
        # 整体平均和历史最高
        history_mean_fitness = []
        history_best_fitness = []
        print("\niter start")
        for i in range(self.max_iter):

            if len(history_best_fitness)>10 and np.var(history_best_fitness[-10:])<1e-9:
                print("\nrestart")
                best_avg_fitness = np.min(history_mean_fitness[-10:])
                for k in range(self.pop_size):
                    if np.random.rand()>=0.5:
                        pop[k]=np.round(np.random.rand(1,dimensions),4)
                        idvl_denorm=min_b+pop[k]*diff
                        fitness[k]=se.run_idvl(idvl_denorm)


                history_best_fitness = []
                history_mean_fitness = []

            self.current_gen = i
            current_generation_fitness = []
            selected_indices = np.random.choice(range(self.pop_size), int(self.pop_size * np.random.uniform(0.7, 1)),
                                                replace=False)

            pop = np.round(pop,4)
            if self.eval_selector == EvalSelect.Single:

                for j in selected_indices:
                    idxs = [idx for idx in range(self.pop_size) if idx != j]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + np.random.uniform(self.mutation, 1) * (b - c)
                    mutant = np.where(mutant < 0, 0, mutant)
                    mutant = np.where(mutant > 1, 1, mutant)
                    cross_points = np.random.rand(dimensions) < self.crossover
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, dimensions)] = True
                    trial = np.where(cross_points, mutant, pop[j])
                    trial = np.round(trial, 4)
                    trial_X = min_b + trial * diff

                    trial_f = se.run_idvl(trial_X)
                    current_generation_fitness.append(trial_f)
                    if trial_f < fitness[j]:
                        fitness[j] = trial_f
                        pop[j] = trial
                        if trial_f < best_f:
                            best_x = trial_X
                            best_f=trial_f

            else:# multi indvl run mode
                input_env = [self.get_DE_multiArgs(j,pop,min_b,max_b,diff,fitness) for j in selected_indices]
                results = self.p.map(sub_process_DE_pop_eval, input_env)
                for trial_f,trial_X,trial, idvl_idx in results:
                    current_generation_fitness.append(trial_f)
                    if trial_f < fitness[idvl_idx]:
                        fitness[idvl_idx] = trial_f
                        pop[idvl_idx] = trial
                        if trial_f < best_f:
                            best_f = trial_f
                            best_x = trial_X

            history_mean_fitness.append(np.mean(current_generation_fitness))
            history_best_fitness.append(best_f)
            yield best_x, best_f, pop, fitness

    def GA(self):
        def selection(population, scores, k=5):
            selection_ix = np.random.randint(len(population), size=k)
            ix = selection_ix[np.argmin(scores[selection_ix])]
            return population[ix]

        def mutation(offspring):
            for i in range(len(offspring)):
                # 判断是否进行变异
                if np.random.rand() < self.mutation/3:
                    # 随机选择变异点
                    new_value = np.round(np.random.uniform(min_b[0], max_b[0], 1),4)[0]
                    offspring[i]= new_value
            return offspring
        def crossover(p1, p2, r_cross):
            # 子代初始化为父代
            c1, c2 = p1.copy(), p2.copy()
            # 判断是否进行交叉
            if np.random.rand() < r_cross:
                # 随机选择交叉点
                pt = np.random.randint(1, len(p1) - 2)
                # 进行交叉
                c1 = np.hstack((p1[:pt], p2[pt:]))
                c2 = np.hstack((p2[:pt], p1[pt:]))
            return [c1, c2]



        min_b, max_b = np.asarray(self.bounds).T
        # 变异操作
        pop = np.round(np.random.uniform(min_b, max_b, (self.pop_size,self.total_param_num)), 4)
        scores = np.array([self.idvl_eval(i) for i in pop])
        g = scores_evaluator(self.data_set_name, self.data_sample_scale, self.random_ratio, self.eval_run_count, self.algo_name)

        best_idx = np.argmin(scores)
        history_best_x = pop[best_idx]

        history_best_f = scores[best_idx]

        for gen in range(self.max_iter):
            # 评估种群
            self.current_gen=gen
            selected_indices = np.random.choice(range(self.pop_size), int(self.pop_size * np.random.uniform(0.7, 1)),
                                                replace=False)

            new_pop = np.zeros((self.pop_size*2, self.total_param_num))
            for i in range(int(self.pop_size)):
                p1 = selection(pop, scores)
                p2 = selection(pop, scores)
                offspring = crossover(p1, p2, self.crossover/3)

                new_pop[i]=mutation(offspring[0])
                new_pop[self.pop_size+i]=mutation(offspring[1])

            new_scores = np.array(self.p.map(g.run_pop, new_pop))
            new_idx = np.argsort(new_scores)[:self.pop_size]
            pop = new_pop[new_idx]
            scores = new_scores[new_idx]
            best_ix = np.argmin(scores)
            best = pop[best_ix]
            if scores[best_ix] < history_best_f:
                history_best_x = best
                history_best_f = scores[best_ix]

            yield history_best_x, history_best_f, pop, scores



    def get_DE_multiArgs(self,idx,pop,min_b,max_b,diff,fitness):
        args = DE_MultiArgs(
                data_set_name=self.data_set_name,
                idvl_idx=idx,
                mutation=self.mutation,
                crossover=self.crossover,
                pop_size=self.pop_size,
                pop=pop,
                min_b=min_b,
                max_b = max_b,
                diff=diff,
                dimensions=self.total_param_num,
                data_sample_scale =self.data_sample_scale,
                random_ratio =self.random_ratio,
                eval_run_count=self.eval_run_count,
                algo_name=self.algo_name,
                pop_fitness=fitness,
        )
        return args
    #
    # def idvl_eval(self, weights: np.ndarray):
    #     """ 这个代码用于专门执行单一的评估操作 用map来并行single_eval"""
    #     start = time()
    #
    #     datas = [SingleAlgoArgs(weights=weights,
    #                             algo_name=self.algo_name,
    #                             data_sample_scale=self.data_sample_scale,
    #                             data_set_name = self.data_set_name,
    #                             random_ratio=self.random_ratio)
    #              for i in range(self.eval_run_count)]
    #     result = self.p.map(sub_process_idvl_eval, datas)
    #     result = np.array(result)
    #     # print(result)
    #     mean = np.mean(result)
    #     std = np.std(result)
    #     cutoff = std * 3
    #     lower, upper = mean - cutoff, mean + cutoff
    #     result = result[(result > lower) & (result < upper)]
    #     mean = np.mean(result)
    #     end = time()
    #     print(f"{round(end - start, 2)}s,{round(mean * 100, 2)}%", end=", ")
    #     gc.collect()
    #     # show_memory()
    #     return 1 / mean


def solution_draw(solution: BinPacking2DAlgo.Algo, text=""):
    for i in range(len(solution.packinglog) - 1, -1, -1):
        new_solution = []
        plan_log = solution.packinglog[i]
        for j in range(len(plan_log)):
            plan = ProtoPlan(i, Rect(0, 0, *MATERIAL_SIZE),
                             [Item(i, Rect(*item), POS(item[0], item[1])) for item in plan_log[j][0]],
                             remain_containers=[Container(POS(rect[0], rect[1]), POS(rect[2], rect[3])) for rect in
                                                plan_log[j][1]])
            new_solution.append(plan)
        standard_draw_plan(new_solution, is_debug=True, task_id=solution.task_id, text=text)


class Training:

    def __init__(self, data_set, training_type: "list" = (STANDARD,), algo_name: "list" = (AlgoName.Dist_MaxRect,)):
        """

        :param training_type: "determ","noised"
        """
        self.training_types = training_type
        self.data_sets = data_set
        self.algo_names = algo_name


    def run(self):
        start_time = time()
        with multiprocessing.Pool() as p:
            result = []
            for data, name in self.data_sets:
                for algo_name in self.algo_names:
                    for training_type in self.training_types:
                        start_time2 = time()
                        print(training_type, name, algo_name, "start")
                        d = Optimizer(p, data, name, random_ratio=(0, 0.3) if training_type == NOISED else None,
                                      eval_selector=EvalSelect.Multi,
                                      algo_name=algo_name)
                        d.run_v2(d.DE)

                    # result.append([name, x, 1 / fun, f"训练用时(秒):{time() - start_time2}"])
                    # np.save(f"{self.training_type}_Dist_{name}_{fun}__{round(time())}.npy", np.array(x))
        end_time = time()
        print("全部训练完成时间(秒):", end_time - start_time)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    t = Training([
        [华为杯_data, PRODUCTION_DATA1],
        # [外包_data, PRODUCTION_DATA2],
        # [随机_data, RANDOMGEN_DATA]
    ],
        training_type=[STANDARD],
        algo_name=[AlgoName.Dist_MaxRect]
    )
    t.run()
    # print(params[STANDARD][PRODUCTION_DATA2])
