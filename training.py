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

def sub_process_pop_eval(arg:DE_MultiArgs):
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

class DE:
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
        self.bounds = [[-self.total_param_num, self.total_param_num]] * self.total_param_num
        self.mutation = 0.4
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

    def run_v2(self):
        self.time_recorder.append(time())
        # current_best = None

        best_score: "np.ndarray|None" = None
        final_best_x: "np.ndarray|None" = None
        for best_x, best_fitness, avg_fitness in self.optimizing():
            self.time_recorder.append((time()))
            if best_score is None:
                best_score = np.array([])
            print(
                f"\ngen={self.current_gen},time_use={round(self.time_recorder[-1] - self.time_recorder[-2], 2)}s,avg_score={round(1 / avg_fitness * 100, 3)}%,hist_best_score={round(1 / best_fitness * 100, 3)}%,x={list(best_x)}")
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

    def optimizing(self):
        print("optimizer init")
        dimensions = self.total_param_num
        pop = np.random.rand(self.pop_size, dimensions)
        min_b, max_b = np.asarray(self.bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([self.idvl_eval(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        # 整体平均和历史最高
        history_mean_fitness = []
        history_best_fitness = []
        print("\niter start")
        for i in range(self.max_iter):

            if len(history_best_fitness)>20 and np.var(history_best_fitness[-20:])<1e-7:
                print("\nrestart")
                best_avg_fitness = np.min(history_mean_fitness[-20:])
                for k in range(self.pop_size):
                    if fitness[k]>best_avg_fitness:
                        pop[k]=np.random.rand(1,dimensions)
                        idvl_denorm=min_b+pop[k]*diff
                        fitness[k]=self.idvl_eval(idvl_denorm)
                history_best_fitness = []
                history_mean_fitness = []

            self.current_gen = i
            current_generation_fitness = []
            selected_indices = np.random.choice(range(self.pop_size), int(self.pop_size * np.random.uniform(0.7, 1)),
                                                replace=False)


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
                    trial_denorm = min_b + trial * diff
                    f = self.idvl_eval(trial_denorm)
                    current_generation_fitness.append(f)
                    if f < fitness[j]:
                        fitness[j] = f
                        pop[j] = trial
                        if f < fitness[best_idx]:
                            best_idx = j
                            best = trial_denorm

            else:# multi indvl run mode
                input_env = [self.get_DE_multiArgs(j,pop,min_b,max_b,diff,fitness) for j in selected_indices]
                results = self.p.map(sub_process_pop_eval, input_env)
                for trial_f,trial_denorm,trial, idvl_idx in results:
                    current_generation_fitness.append(trial_f)
                    if trial_f < fitness[idvl_idx]:
                        fitness[idvl_idx] = trial_f
                        pop[idvl_idx] = trial
                        if trial_f < fitness[best_idx]:
                            best_idx = idvl_idx
                            best = trial_denorm
            history_mean_fitness.append(np.mean(current_generation_fitness))
            history_best_fitness.append(fitness[best_idx])

            yield best, fitness[best_idx], history_mean_fitness[-1]

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

    def get_sampled_items(self) -> np.ndarray:

        if self.random_ratio is not None:
            return self.random_mix()
        else:
            return kde_sample(self.data_set, self.data_sample_scale)

    def random_mix(self):
        determ_data = kde_sample(self.data_set, self.data_sample_scale)[:, 1:]
        random_item_scale = int(np.random.uniform(*self.random_ratio) * self.data_sample_scale)
        determ_data_idx = np.random.choice(self.data_sample_scale, size=self.data_sample_scale - random_item_scale,
                                           replace=False)
        determ_data = determ_data[determ_data_idx]
        random_x = (np.random.uniform(0.1, 1, random_item_scale) * MATERIAL_SIZE[0]).astype(int)
        random_y = (np.random.uniform(0.1, 1, random_item_scale) * MATERIAL_SIZE[1]).astype(int)
        random_data = np.column_stack((random_x, random_y))
        result = np.row_stack((determ_data, random_data))
        result = np.column_stack((range(self.data_sample_scale), result))
        return result

    def idvl_eval(self, weights: np.ndarray):
        """ 这个代码用于专门执行单一的评估操作 用map来并行single_eval"""
        start = time()

        datas = [SingleAlgoArgs(weights=weights,
                                algo_name=self.algo_name,
                                data_sample_scale=self.data_sample_scale,
                                data_set_name = self.data_set_name,
                                random_ratio=self.random_ratio)
                 for i in range(self.eval_run_count)]
        result = self.p.map(sub_process_idvl_eval, datas)
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
        gc.collect()
        # show_memory()
        return 1 / mean


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
                        d = DE(p,data, name, random_ratio=(0, 0.3) if training_type == NOISED else None,
                               eval_selector=EvalSelect.Multi,
                               algo_name=algo_name)
                        d.run_v2()

                    # result.append([name, x, 1 / fun, f"训练用时(秒):{time() - start_time2}"])
                    # np.save(f"{self.training_type}_Dist_{name}_{fun}__{round(time())}.npy", np.array(x))
        end_time = time()
        print("全部训练完成时间(秒):", end_time - start_time)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    t = Training([
        # [华为杯_data, PRODUCTION_DATA1],
        # [外包_data, PRODUCTION_DATA2],
        [随机_data, RANDOMGEN_DATA]
    ],
        training_type=[STANDARD],
        algo_name=[AlgoName.Dist_Skyline]
    )
    t.run()
    # print(params[STANDARD][PRODUCTION_DATA2])
