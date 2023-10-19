# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'net.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/7 15:47'
"""
import multiprocessing
import pickle
# 对于评分函数 提供三种神经网络,第一种是SLP,10个输入, 第二,三种是 MLP,多层感知机, 层数分别为 10*4*4*1和10*4*4*4*4*1
# 对于排序函数 提供两种神经网络,第一种是SLP,6个输入,第二种是SLP,6*4*4*1
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
        MLP = 0
        GA = 1

    def __init__(self, algo: "Distribution"):
        self.version = self.V.GA
        self.container_scoring_arch = np.array([10, 10, 8, 4, 1])
        self.item_sorting_arch = np.array([6, 6, 4, 1])
        self.algo = algo
        self.parameters = np.ones(self.total_param_num)
        # self.container_scoring_parameters = self.parameters[:self.container_scoring_param_count]
        # self.item_sorting_parameters:"np.ndarray" = self.parameters[self.container_scoring_param_count:]
        # self.total_param_num=self.item_sorting_param_count+self.container_scoring_param_count
        # self.algo:"Algo"=algo
        # self.plans=plans
        # self.choosed_container=container

    @property
    def container_scoring_parameters(self):
        return self.parameters[:self.container_scoring_param_count]

    @property
    def item_sorting_parameters(self):
        # print("parameters_total=",self.parameters,"item_sorting_parameters=",self.parameters[self.container_scoring_param_count:],"self.container_scoring_param_count=",self.container_scoring_param_count)
        return self.parameters[self.container_scoring_param_count:]

    @property
    def total_param_num(self):
        return self.item_sorting_param_count + self.container_scoring_param_count

    @property
    def container_scoring_param_count(self):
        if self.version == self.V.GA:
            return 14
        else:
            return sum((x + 1) * y for x, y in zip(self.container_scoring_arch, self.container_scoring_arch[1:] + [0]))

    @property
    def item_sorting_param_count(self):
        if self.version == self.V.GA:
            return 4
        else:
            return sum((x + 1) * y for x, y in zip(self.item_sorting_arch, self.item_sorting_arch[1:] + [0]))

    def item_sorting(self, item_width, item_height) -> float | int:
        if self.version == self.V.MLP:
            X = np.array([item_width, item_height, self.algo.maxL, self.algo.minL, self.algo.material.width, self.algo.material.height])
            return self.multi_layer_perceptron(X, self.item_sorting_parameters, self.item_sorting_arch)
        else:
            X = np.array([
                    (item_width * item_height) / (self.algo.minL * self.algo.maxL),  # item area
                    item_height / item_width,  # side ratio
                    (item_width * item_height) / self.algo.material.area,
                    abs(item_width - item_height) / (self.algo.maxL - self.algo.minL),
            ])
            return np.dot(X, self.item_sorting_parameters)

    def pos_scoring(self, item_width, item_height, container_begin_x, container_begin_y, container_width, container_height, plan_id) -> float | int:
        if self.version == self.V.MLP:
            rate = self.algo.solution[plan_id].util_rate() if plan_id > -1 else 0
            X = np.array([item_width, item_height, container_begin_x, container_begin_y, container_width, container_height, self.algo.material.width, self.algo.material.height, plan_id, rate])
            return self.multi_layer_perceptron(X, self.container_scoring_parameters, self.container_scoring_arch)
        else:
            X = np.array([
                    (item_width * item_height) / (self.algo.minL * self.algo.maxL),  # item area
                    item_height / item_width,  # side ratio
                    (item_width * item_height) / self.algo.material.area,
                    abs(item_width - item_height) / (self.algo.maxL - self.algo.minL),
                    (plan_id + 1) / len(self.algo.solution) if self.algo.solution else 0,
                    (item_width * item_height) / (container_width * container_height),
                    1 - (item_width * item_height) / (container_width * container_height),
                    1 - item_width / container_width,
                    1 - item_height / container_height,
                    (container_width * container_height) / self.algo.material.area,
                    container_begin_x / self.algo.material.width,
                    container_begin_y / self.algo.material.height,
                    self.algo.material.height / self.algo.material.width,
                    self.algo.solution[plan_id].util_rate() if plan_id >= 0 else 0,
            ])
            return np.dot(X, self.container_scoring_parameters)

    def can_container_merge(self, c1, c2, maxL, minL, plan_id, ):
        pass

    @staticmethod
    def multi_layer_perceptron(X, parameters, layer_dims):
        # print("MLP Parameter=",parameters)
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

    def __init__(self, algo: "Distribution", item: Item, container: Container, plan_id: int):
        self.item = item
        self.container = container
        self.plan_id = plan_id
        self.algo: Distribution = algo

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
    def __init__(self, item_data: "np.ndarray", material_data: "Iterable" = MATERIAL_SIZE, task_id=None):
        super().__init__(item_data, material_data, task_id)
        # self.parameters = parameters
        # self.parameters=parameters
        self.scoring_sys = ScoringSys(self)

        self.maxL = max(self.items, key=lambda x: x.size.width).size.width
        self.minL = min(self.items, key=lambda x: x.size.height).size.height

    def run(self, is_debug=False):
        self.solution = []
        min_item_length = min(self.items, key=lambda x: x.size.height)

        sorted_items = self.sorted_items()

        for at_i in range(len(sorted_items)):

            new_item: Item = sorted_items[at_i]

            # print("current item count = ",item_remain)
            pre_score_list: list[ItemScore] = []
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

                    if new_item.size + corner_start in container:
                        item_to_append = new_item.copy()
                        item_to_append.pos = corner_start
                        pre_score_list.append(
                                ItemScore(
                                        item=item_to_append,
                                        # material=self.material,
                                        container=container,
                                        plan_id=plan.ID,
                                        algo=self,
                                        # plans=plans,
                                        # scoring_system = self.scoring_sys
                                )
                        )
                    itemT = new_item.transpose()
                    if itemT.size + corner_start in container:
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
                fake_container = Container(POS(0, 0), POS(*MATERIAL_SIZE))
                corner_start = fake_container.rect.start
                if new_item.size + corner_start in fake_container:
                    # 处理新板材
                    pre_score_list.append(
                            ItemScore(
                                    item=new_item.copy(),
                                    algo=self,
                                    container=Container(POS(0, 0), POS(self.material.width, self.material.height)),
                                    plan_id=-1,
                                    # plans=plans,
                                    # scoring_system=self.scoring_sys
                            )
                    )

                itemT = new_item.transpose()
                if itemT.size + corner_start in fake_container:
                    pre_score_list.append(
                            ItemScore(
                                    item=itemT.copy(),
                                    # material=self.material,
                                    container=Container(POS(0, 0), POS(self.material.width, self.material.height)),
                                    plan_id=-1,
                                    algo=self,
                                    # plans=plans,
                                    # scoring_system=self.scoring_sys
                            )
                    )

            if len(pre_score_list) == 0:
                raise ValueError("pre_score_list is empty", new_item)

            bestscore: "ItemScore" = max(pre_score_list, key=lambda x: x.score)

            new_rect = bestscore.item.rect
            remove_rect = bestscore.container.rect
            new_BR_corner = Container(new_rect.bottomRight,
                                      POS(remove_rect.topRight.x, new_rect.topRight.y),
                                      bestscore.plan_id)
            new_top_corner = Container(new_rect.topLeft, remove_rect.topRight, bestscore.plan_id)
            current_minL = min(sorted_items[at_i:], key=lambda x: min(x.size.height, x.size.width)).size.height
            current_maxL = max(sorted_items[at_i:], key=lambda x: max(x.size.height, x.size.width)).size.width
            if new_BR_corner.rect.end.y - new_BR_corner.rect.start.y < current_minL:
                new_BR_corner = None
            if new_top_corner.rect.end.y - new_top_corner.rect.start.y < current_minL:
                if new_BR_corner is not None:
                    new_BR_corner.rect.end = new_top_corner.rect.end
                    new_top_corner = None

            # if new_material_id >= 0:  # 新建
            if bestscore.plan_id == -1:
                # print("add to new plan")
                # idx = np.where(std_materials[:, COL.ID] == new_material_id)[0][0]
                # std_materials[idx][COL.Remain] -= 1
                # X, Y = get_material_XY(new_material_id, std_materials)
                X, Y = self.material.size
                # X, Y = float(X), float(Y)
                # legal_zone = X - RIGHT, Y - TOP
                # new_corner_plan = [new_material_id]
                new_plan = ProtoPlan(len(self.solution), self.material.copy(), [], [])
                new_plan.item_sequence.append(bestscore.item)
                if is_debug:
                    standard_draw_plan([new_plan], is_debug=is_debug, task_id=self.task_id, text=f"插入矩形{bestscore.item.rect}")
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
                        new_BR_corner = self.container_merge_thinking(is_debug,current_minL,current_maxL,plan,new_BR_corner,container)

                    if new_top_corner is not None:
                        new_top_corner = self.container_merge_thinking(is_debug,current_minL,current_maxL,plan,new_top_corner,container)

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

    def container_merge_thinking(self,is_debug,current_minL,current_maxL,plan:"ProtoPlan",compare_corner:"Container",container:"Container")->Container|None:
        result = container.rect & compare_corner.rect
        merged = False
        if result == Line:  # 相切则合并
            diff = result.end - result.start
            if diff.x == 0:  # 垂直
                # 新的在右边
                if compare_corner.rect.bottomLeft == container.rect.bottomRight and compare_corner.rect.topLeft == container.rect.topRight:
                    container.rect.end = compare_corner.rect.end.copy()
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并{container.rect}")
                    merged = True
                # 新的在左边
                elif compare_corner.rect.bottomRight == container.rect.bottomLeft and compare_corner.rect.topRight == container.rect.topLeft:
                    container.rect.start = compare_corner.rect.start.copy()
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并{container.rect}")
                    merged = True
            elif diff.y == 0:  # 水平
                # 新的在上面
                if compare_corner.rect.bottomRight == container.rect.topRight and compare_corner.rect.bottomLeft == container.rect.topLeft:
                    container.rect.end = compare_corner.rect.end.copy()
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并{container.rect}")
                    merged = True
                # 新的在下面
                elif compare_corner.rect.topRight == container.rect.bottomRight and compare_corner.rect.topLeft == container.rect.bottomLeft:
                    container.rect.start = compare_corner.rect.start.copy()
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并{container.rect}")
                    merged = True
                # 新的在下面 右侧相等,左侧略有偏差
                elif compare_corner.rect.topRight == container.rect.bottomRight and \
                        abs(compare_corner.rect.topLeft.x - container.rect.bottomLeft.x) < current_minL :
                    container.rect.start = POS(
                            max(container.rect.start.x, compare_corner.rect.start.x),
                            compare_corner.rect.start.y)
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并{container.rect}")
                    merged = True

                # 新的在上面 右侧相等,左侧略有偏差
                elif compare_corner.rect.bottomRight == container.rect.topRight and \
                        abs(compare_corner.rect.bottomLeft.x - container.rect.topLeft.x) < current_minL :
                    container.rect.end = compare_corner.rect.end.copy()
                    container.rect.start = POS(
                            max(container.rect.topLeft.x, compare_corner.rect.bottomLeft.x),
                            container.rect.start.y
                    )
                    if is_debug:
                        standard_draw_plan([plan], is_debug=is_debug, task_id=self.task_id, text=f"容器合并{container.rect}")
                    merged = True
        return None if merged else compare_corner


    def sorted_items(self):
        temp_sorted_items = sorted(self.items, key=lambda x: self.scoring_sys.item_sorting(x.size.width, x.size.height))
        return temp_sorted_items

    def eval(self, parameters, count=18):
        # print("eval_parameters ",parameters)
        start_time = time()
        self.scoring_sys.parameters = parameters
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
        # print("EVAL END")
        value = 1 / np.average(result)
        end_time = time()
        print(f"value={value},time cost = {end_time - start_time}")
        return value,

    def fit_DE(self, scoring_version=ScoringSys.V.GA):
        start_time = time()
        self.scoring_sys.version = scoring_version
        from scipy.optimize import differential_evolution
        time_recorder = [time()]
        traning_log = []

        def callback(xk, convergence):
            time_recorder.append(time())
            traning_log.append([len(time_recorder), 1 / self.eval(xk)[0]])
            print(f'time cost {time_recorder[-1] - time_recorder[-2]}Current solution: {xk}, Convergence: {convergence}')

        # init = None
        bounds = [[-20, 20]] * self.scoring_sys.total_param_num
        result = differential_evolution(self.eval, bounds, workers=-1, atol=0.0001, strategy="randtobest1exp", popsize=12, callback=callback, maxiter=100)
        end_time = time()
        print(end_time - start_time)
        to_save = [end_time, result.x, result.fun]
        print(to_save)
        np.save(f"华为杯_data_traning_log__{end_time}.npy", np.array(traning_log))
        return result.x, result.fun, traning_log













if __name__ == "__main__":
    import matplotlib as plt

    print("start")
    start_time = time()

    np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(华为杯_data.shape[0], 1000)
    data = 华为杯_data[data_idx]
    d = Distribution(data)
    d.scoring_sys.version = ScoringSys.V.GA
    # print(list(best_ind),best_score)
    d.scoring_sys.parameters = [-12.764467729922428, -7.2807524490032804, -17.405272153673526, 11.62060943355495, -17.767676767373285, 13.498788968865574, -3.058679224306764, -17.380930383866435, -17.380008727391687, -19.579085902347263, 15.561194939767207, 2.310615782862815, -5.273339286206582, 1.6631169187587558, -1.906345802422087, -3.3207320056750733, -7.4098035553284936, 12.394940621852495] # 1.07
    print(d.task_id)
    # d.run(is_debug=True)
    d.run(is_debug=False)
    # task=bcafba05 id=1
    # c1 =Container(POS(2349,653),POS(2440,853))
    # c2 =Container(POS(2370,853),POS(2440,1176))
    # d.container_merge_thinking(False,100,1000,None,c1,c2)

    end_time = time()

    print("time cost=", end_time - start_time,"rate=",d.avg_util_rate())
    standard_draw_plan(d.solution, task_id=d.task_id)

    # new_BR_corner = Container(POS(1000,200),POS(1300,400))
    # container = Container(POS(1200))
    # 新的在上面
    # if new_BR_corner.rect.bottomRight == container.rect.topRight and abs(new_BR_corner.rect.start.x - container.rect.topLeft.x) < current_minL * 2:
    #     container.rect.end = new_BR_corner.rect.end.copy()
    #     container.rect.start = POS(
    #             max(container.rect.start.x, new_BR_corner.rect.start.x),
    #             container.rect.start.y
    #     )

    pass
