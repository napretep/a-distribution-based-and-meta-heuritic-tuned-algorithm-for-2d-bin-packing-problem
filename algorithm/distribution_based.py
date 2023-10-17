# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'net.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/7 15:47'
"""
#对于评分函数 提供三种神经网络,第一种是SLP,10个输入, 第二,三种是 MLP,多层感知机, 层数分别为 10*4*4*1和10*4*4*4*4*1
#对于排序函数 提供两种神经网络,第一种是SLP,6个输入,第二种是SLP,6*4*4*1
from dataclasses import dataclass
from time import time

import numpy as np
from constant import *
from enum import Enum, auto
from visualizing.draw_plan import standard_draw_plan

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 感知机模型和训练
# 训练后保存参数,使用时读取参数
class ScoringSys:
    class V:
        MLP=0
        GA=1
    def __init__(self,algo:"Distribution"):
        self.version=self.V.GA
        self.pos_scoring_arch=np.array([10, 10, 8, 4, 1])
        self.item_sorting_arch=np.array([6, 6, 4, 1])
        self.algo=algo
        self.container_scoring_param_count =14 #sum((x + 1) * y for x,y in zip(self.pos_scoring_arch, self.pos_scoring_arch[1:] + [0]))
        self.item_sorting_param_count =4 #sum((x+1)*y for x,y in zip(self.item_sorting_arch, self.item_sorting_arch[1:] + [0]))
        self.parameters=self.algo.parameters if self.algo.parameters else np.ones(self.item_sorting_param_count + self.container_scoring_param_count)
        self.pos_scoring_parameters = self.parameters[:self.container_scoring_param_count]
        self.item_sorting_parameters:"np.ndarray" = self.parameters[self.container_scoring_param_count:]
        self.total_param_num=self.item_sorting_param_count+self.container_scoring_param_count
        # self.algo:"Algo"=algo
        # self.plans=plans
        # self.choosed_container=container




    def item_sorting(self, item_width, item_height)-> float | int:
        X = np.array([
                (item_width * item_height)/(self.algo.minL*self.algo.maxL), # item area
                item_height/item_width,  # side ratio
                (item_width * item_height) / self.algo.material.area,
                abs(item_width - item_height)/(self.algo.maxL-self.algo.minL),
        ])
        return np.dot(X, self.item_sorting_parameters)
        # X = np.array([item_width,item_height,self.algo.maxL,self.algo.minL,self.algo.material.width,self.algo.material.height])
        # return self.multi_layer_perceptron(X, self.sorting_parameters, self.item_sorting_arch)



    def pos_scoring(self, item_width, item_height, container_begin_x, container_begin_y, container_width, container_height, plan_id)-> float | int:
        # rate = self.algo.solution[plan_id].util_rate() if plan_id> -1 else 0
        # X = np.array([item_width,item_height,container_begin_x,container_begin_y,container_width,container_height,self.algo.material.width,self.algo.material.height,plan_id,rate])
        # return self.multi_layer_perceptron(X, self.pos_scoring_parameters, self.pos_scoring_arch)
        X = np.array([
                (item_width * item_height) / (self.algo.minL * self.algo.maxL),  # item area
                item_height/item_width ,  # side ratio
                (item_width * item_height) / self.algo.material.area,
                abs(item_width - item_height) / (self.algo.maxL - self.algo.minL),
                (plan_id+1)/len(self.algo.solution) if self.algo.solution else 0,
                (item_width * item_height) / (container_width * container_height),
                1-(item_width * item_height) / (container_width * container_height),
                1-item_width/container_width,
                1-item_height/container_height,
                (container_width * container_height)/self.algo.material.area,
                container_begin_x/self.algo.material.width,
                container_begin_y/self.algo.material.height,
                self.algo.material.height/self.algo.material.width,
                self.algo.solution[plan_id].util_rate() if plan_id>=0 else 0,
        ])
        return np.dot(X,self.pos_scoring_parameters)


    @staticmethod
    def multi_layer_perceptron(X, parameters, layer_dims):
        start = 0
        A = X.copy()
        # print(X.shape,parameters.shape,layer_dims)
        for l in range(1, len(layer_dims)):
            end = start + layer_dims[l] * layer_dims[l - 1]
            W = parameters[start:end].reshape(layer_dims[l], layer_dims[l - 1])
            start = end
            end = start + layer_dims[l]
            b = parameters[start:end].reshape(layer_dims[l], 1)
            start = end
            A_prev = A
            A = sigmoid(np.dot(W, A_prev) + b)
        # print(A[0][0])
        return A[0][0]

@dataclass
class ItemScore:
    item: Item
    # material:"Rect"
    container: Container
    # plans:"list[ProtoPlan]"
    plan_id: int
    # scoring_system: "ScoringSys"

    def __init__(self,algo:"Distribution",item:Item,container:Container,plan_id:int):
        self.item=item
        self.container=container
        self.plan_id=plan_id
        self.algo:Distribution=algo

    @property
    def score(self):
        # rate = 0 if self.plan_id == -1 else self.algo.solution[self.plan_id].util_rate()
        s = self.algo.scoring_sys.pos_scoring(
                self.item.size.width,
                self.item.size.height,
                self.container.rect.start.x,
                self.container.rect.start.y,
                self.container.rect.width,
                self.container.rect.height,
                self.plan_id
        )
        return s

class Distribution(Algo):
    def __init__(self,item_data: "np.ndarray", material_data: "Iterable" = MATERIAL_SIZE, task_id=None,parameters=None):
        super().__init__(item_data, material_data,task_id)
        # self.parameters = parameters
        self.parameters=parameters
        self.scoring_sys = ScoringSys(self)

        self.maxL = max(self.items, key=lambda x: x.size.width).size.width
        self.minL = min(self.items, key=lambda x: x.size.height).size.height
    def run(self,is_debug=False):
        # std_items = self.items
        # std_materials = self.material
        self.solution=[]
        min_item_length = min(self.items, key=lambda x: x.size.height)

        # plan_updated_for_debug = []

        std_items = self.sorted_items()

        for at_i in range(len(std_items)):
        # for at_i in range(std_items.shape[0]):

            new_item:Item = std_items[at_i]

            # print("current item count = ",item_remain)
            pre_score_list:list[ItemScore] = []
            # new_material_candidates =
            # print("new_material_candidates=",new_material_candidates)
            # 处理旧板材
            # for idx in range(len(corners_in_planed_materials)):
            for plan in self.solution:
                # corners = corners_in_planed_materials[idx]
                containers = plan.remain_containers
                # material_id = corners[0]
                X, Y = self.material  # std_materials[std_materials[:,COL.Material.ID]==material_id][0,:COL.minL+1]
                # legal_size = X - RIGHT, Y - TOP
                # for corner in corners[1:]:
                for container in containers:
                    (corner_start, corner_end) = container.rect
                    # m_maxL, m_minL = x2-del_corner_x0,y2-del_corner_y0

                    if new_item.size+corner_start in container:
                        item_to_append = new_item.copy()
                        item_to_append.pos=corner_start
                        pre_score_list.append(
                                ItemScore(
                                    item = item_to_append,
                                    # material=self.material,
                                    container = container,
                                    plan_id = plan.ID,
                                    algo=self,
                                    # plans=plans,
                                    # scoring_system = self.scoring_sys
                                )
                        )
                    itemT = new_item.transpose()
                    if itemT.size+corner_start in container:
                        item_to_append = itemT.copy()
                        item_to_append.pos = corner_start
                        pre_score_list.append(
                                ItemScore(
                                        item=item_to_append,
                                        # material=self.material,
                                        container=container,
                                        plan_id=plan.ID,
                                        algo=self,
                                        # plans=plans,
                                        # scoring_system=self.scoring_sys
                                )
                        )
                # if len(pre_score_list)>0:
                #     break

            if len(pre_score_list) == 0:
                # 处理新板材
                pre_score_list.append(
                        ItemScore(
                                item=new_item,
                                algo=self,
                                container=Container(POS(0, 0), POS(self.material.width, self.material.height)),
                                plan_id=-1,
                                # plans=plans,
                                # scoring_system=self.scoring_sys
                        )
                )
                pre_score_list.append(
                        ItemScore(
                                item=new_item.transpose(),
                                # material=self.material,
                                container=Container(POS(0, 0), POS(self.material.width, self.material.height)),
                                plan_id=-1,
                                algo=self,
                                # plans=plans,
                                # scoring_system=self.scoring_sys
                        )
                )

            if len(pre_score_list)==0:
                raise ValueError("pre_score_list is empty",new_item)
            # 评分,获取方案
            # if pre_score_list:
            bestscore:"ItemScore" = max(pre_score_list, key=lambda x: x.score)
            min_gap = min_item_length
            # max_score = mainFun_get_best_score_plan(pre_score_list, plan_for_cutting, cutting_gap, std_materials, score_parameters, score_calculators)
            # score, new_material_id, plan_id, item_id, (itemPos_start, itemPos_end), (delCorner_start, delCorner_end) = max_score

            # cut_pos = itemPos_end[x] + cutting_gap, \
            #           itemPos_end[y] + cutting_gap
            # new_BR_corner = (cut_pos[x], delCorner_start[y]), (delCorner_end[x], cut_pos[y])
            # new_top_corner = (delCorner_start[x], cut_pos[y]), delCorner_end
            new_rect = bestscore.item.rect
            remove_rect = bestscore.container.rect
            new_BR_corner = Container(new_rect.bottomRight,
                                      POS(remove_rect.topRight.x, new_rect.topRight.y),
                                      bestscore.plan_id)
            new_top_corner = Container(new_rect.topLeft,remove_rect.topRight,bestscore.plan_id)
            if new_BR_corner.rect.end.x-new_BR_corner.rect.start.x < self.minL:
                new_BR_corner = None
            if new_top_corner.rect.end.x-new_top_corner.rect.start.x < self.minL:
                if new_BR_corner is not None:
                    new_BR_corner.rect.end=new_top_corner.rect.end
                    new_top_corner = None


            # if new_material_id >= 0:  # 新建
            if bestscore.plan_id == -1:
                # print("add to new plan")
                # idx = np.where(std_materials[:, COL.ID] == new_material_id)[0][0]
                # std_materials[idx][COL.Remain] -= 1
                # X, Y = get_material_XY(new_material_id, std_materials)
                X,Y=self.material.size
                # X, Y = float(X), float(Y)
                # legal_zone = X - RIGHT, Y - TOP
                # new_corner_plan = [new_material_id]
                new_plan = ProtoPlan(len(self.solution),self.material.copy(),[],[])
                new_plan.item_sequence.append(bestscore.item)
                if is_debug:
                    standard_draw_plan([new_plan],is_debug=is_debug,task_id=self.task_id,text="插入矩形")
                if new_BR_corner is not None:
                    if new_BR_corner.rect.start.y >= Y or new_BR_corner.rect.start.x >= X:
                        new_BR_corner = None
                    else:
                        new_plan.remain_containers.append(new_BR_corner)
                        if is_debug:
                            standard_draw_plan([new_plan], is_debug=is_debug, task_id=self.task_id, text=f"插入容器{new_BR_corner.rect}")
                if new_top_corner is not None:
                    if new_top_corner.rect.start.y >= Y or new_top_corner.rect.start.x >= X:
                        new_top_corner = None
                    else:
                        new_plan.remain_containers.append(new_top_corner)
                        if is_debug:
                            standard_draw_plan([new_plan], is_debug=is_debug, task_id=self.task_id, text=f"插入容器{new_top_corner.rect}")
                self.solution.append(new_plan)


            else:  # 原有
                plan = self.solution[bestscore.plan_id]
                plan.remain_containers.remove(bestscore.container)
                if is_debug:
                    standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"移除容器{bestscore.container.rect}")
                plan.item_sequence.append(bestscore.item)
                if is_debug:
                    standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"添加矩形{bestscore.item.rect}")
                # 用来合并一些容器
                for container in plan.remain_containers:
                    if new_BR_corner is None and new_top_corner is None:
                        break
                    if new_BR_corner is not None:
                        result = container.rect&new_BR_corner.rect
                        if result == Line:  # 相切则合并
                            diff = result.end - result.start
                            if diff.x == 0: # 垂直
                                if result.start == container.rect.bottomRight and result.end == container.rect.topRight:
                                    container.rect.end = new_BR_corner.rect.end
                                    new_BR_corner = None
                                elif result.start == container.rect.bottomLeft and result.end == container.rect.topLeft:
                                    container.rect.start = new_BR_corner.rect.start
                                    new_BR_corner = None
                            elif diff.y == 0:# 水平
                                if result.end == container.rect.topRight and result.end == container.rect.topLeft:
                                    container.rect.end = new_BR_corner.rect.end
                                    new_BR_corner = None
                                elif result.end == container.rect.bottomRight and result.end == container.rect.bottomLeft:
                                    container.rect.start = new_BR_corner.rect.start
                                    new_BR_corner = None
                    if new_top_corner is not None:
                        result = container.rect & new_top_corner.rect
                        if result == Line:  # 相切则合并
                            diff = result.end - result.start
                            if diff.x == 0:
                                if result.start == container.rect.bottomRight and result.end == container.rect.topRight:
                                    container.rect.end = new_top_corner.rect.end
                                    new_BR_corner = None
                                elif result.start == container.rect.bottomLeft and result.end == container.rect.topLeft:
                                    container.rect.start = new_top_corner.rect.start
                                    new_BR_corner = None
                            elif diff.y == 0:
                                if result.end == container.rect.topRight and result.end == container.rect.topLeft:
                                    container.rect.end = new_top_corner.rect.end
                                    new_BR_corner = None
                                elif result.end == container.rect.bottomRight and result.end == container.rect.bottomLeft:
                                    container.rect.start = new_top_corner.rect.start
                                    new_BR_corner = None
                if is_debug:
                    standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并")
                if new_BR_corner is not None:
                    plan.remain_containers.append(new_BR_corner)
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"添加容器{new_BR_corner.rect}")
                if new_top_corner is not None:
                    plan.remain_containers.append(new_top_corner)
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"添加容器{new_top_corner.rect}")


            if is_debug:
                standard_draw_plan([self.solution[-1]], is_debug=is_debug, task_id=self.task_id, text=f"最终效果")

        return self.solution
    def sorted_items(self):
        temp_sorted_items = sorted(self.items, key=lambda x:self.scoring_sys.item_sorting(x.size.width, x.size.height))
        return temp_sorted_items


    def eval(self,parameters,count=18):
        self.parameters=parameters
        result = []
        for i in range(count):
            np.random.seed(int(time() * 10000) % 4294967296)
            data_idx = np.random.choice(华为杯_data.shape[0], 300)
            data = 华为杯_data[data_idx]
            self.load_data(data)
            self.run()
            result.append(self.avg_util_rate())
        result.remove(max(result))
        result.remove(min(result))
        return np.average(result),

    def fit_NN_ea(self, pop_size=36, max_gen=100, init_population=None):
        from deap import base, creator, tools, algorithms
        import random
        from multiprocessing import Pool
        print("traning start")
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # 使用并行处理来计算适应度函数
        pool = Pool()
        toolbox.register("map", pool.map)

        # 定义属性（决策变量）的初始化方式
        toolbox.register("attr_float", random.uniform, -100, 100)

        # 自定义一个函数来创建单个个体

        # Structure initializers
        gene_length = self.scoring_sys.total_param_num
        # 定义个体和种群的初始化方式
        if init_population is None:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=gene_length)
        else:
            def create_individual():
                return creator.Individual(init_population)

            toolbox.register("individual", create_individual)
            # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 定义遗传算法的操作
        toolbox.register("evaluate", self.eval)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # 初始化种群
        pop = toolbox.population(n=pop_size)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = "gen", "max"

        for gen in range(max_gen):
            start_time = time()
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(offspring, k=len(pop))
            best_ind = max(pop, key=lambda ind: ind.fitness.values)
            print(f"Best fitness in generation : {best_ind.fitness.values}")
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)

            end_time = time()
            print(end_time - start_time)
            print(gen, "over", )

        best_individual = tools.selBest(pop, k=1)[0]
        print("Best individual is: %s\nwith fitness: %s" % (best_individual, best_individual.fitness))
        #
        # the_best_solution = [None]
        return best_individual, best_individual.fitness,logbook


    def fit_ea(self,pop_size=36, max_gen=100, init_population=None):
        from deap import base, creator, tools, algorithms
        import random
        from multiprocessing import Pool
        print("traning start")
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # 使用并行处理来计算适应度函数
        pool = Pool()
        toolbox.register("map", pool.map)

        # 定义属性（决策变量）的初始化方式
        toolbox.register("attr_float", random.uniform, -2, 2)

        # 自定义一个函数来创建单个个体

        # Structure initializers
        gene_length = self.scoring_sys.total_param_num
        # 定义个体和种群的初始化方式
        if init_population is None:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=gene_length)
        else:
            def create_individual():
                return creator.Individual(init_population)

            toolbox.register("individual", create_individual)
            # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def deap_mutation(individual, toolbox, F=0.5):
            size = len(individual)
            a, b, c = toolbox.select(toolbox.population(n=size), k=3)
            mutant = toolbox.clone(a)
            index = random.randrange(size)
            for i in range(size):
                if i == index or random.random() < F:
                    mutant[i] = a[i] + F * (b[i] - c[i])
            return mutant,

        # 定义遗传算法的操作
        toolbox.register("evaluate", self.eval)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", deap_mutation)
        toolbox.register("select", tools.selBest)
        # 初始化种群
        pop = toolbox.population(n=pop_size)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = "gen", "max"

        for gen in range(max_gen):
            start_time = time()
            offspring = algorithms.eaMuCommaLambda(pop, toolbox, mu=12, lambda_=24, cxpb=0.3, mutpb=0.3,ngen=max_gen)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(offspring,k=pop_size)
            best_ind = max(pop, key=lambda ind: ind.fitness.values)
            print(f"Best fitness in generation {best_ind} : {best_ind.fitness.values}")
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)

            end_time = time()
            print(end_time - start_time)
            print(gen, "over", )

        best_individual = tools.selBest(pop, k=1)[0]
        print("Best individual is: %s\nwith fitness: %s" % (best_individual, best_individual.fitness))
        #
        # the_best_solution = [None]
        return best_individual, best_individual.fitness, logbook


if __name__ == "__main__":
    import matplotlib as plt
    start_time = time()
    init_pop = [0.732621371038817, 0.27689683065424686, -1.5549311561474415, -0.10364913818754412, -0.3183939508251832, -0.1876724590250347, -0.36244424848144485, 0.9332322399113105, 0.10721160370194749, 0.1342500283998984, -0.605439486931415, -0.6854014322480135, -0.9954102772339521, -0.6477256710628161, 0.3420227898005268, -0.10250084110444613, -0.6822676418415798, 0.5947435912790431] # 0.694

    d = Distribution(华为杯_data)
    best_ind,best_score,logbook= d.fit_ea(init_population=init_pop)
    print(best_score)
    print(best_ind)
    gen = logbook.select("gen")
    fit_maxs = logbook.select("max")
    np.save(f"distribution_based_ea_{start_time}.npy",np.array([gen,fit_maxs]))
    # fig, ax1 = plt.subplots()
    # line1 = ax1.plot(gen, fit_maxs, "b-", label="maximum Fitness")
    # ax1.set_xlabel("Generation")
    # ax1.set_ylabel("Fitness", color="b")
    # plt.show()
    # print(best_ind,best_score)

    end_time = time()
    print(end_time-start_time)
    # standard_draw_plan(d.solution,task_id=d.task_id)
    pass