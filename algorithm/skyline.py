# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'skyline.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 17:44'
"""
import uuid
from dataclasses import dataclass, field
from time import time

# - SKYLINE-MW-WM-BFF-DESCSS算法实现
# SKYLINE - 表示使用Skyline数据结构进行打包
# MW - 表示用MinWaste原则来选择空余矩形
# WM - 表示结合使用Waste Map技术优化
# BFF - 表示允许打包到多个箱子中(多箱打包)
# DESCSS - 表示按照矩形的短边长度降序排序。
import numpy as np
from constant import *
from functools import cmp_to_key
from enum import Enum, auto
from visualizing.draw_plan import standard_draw_plan


class ScoreType(Enum):
    WasteMap = auto()
    SkyLine = auto()


@dataclass
class Plan(ProtoPlan):
    skyLineContainers: "list[Container]|None" = field(default_factory=list)  # 此处,freeContainers是有序的list,需要维护和排序
    wasteMap: "list[Container]|None" = field(default_factory=list)

    def get_remain_containers(self):
        return self.skyLineContainers + self.wasteMap


@dataclass
class ItemScore:
    item: Item
    container_range: tuple[int, int]
    score: tuple[int | float, int | float] | int | float  # 先比较浪费空间谁最小,然后比较min_y谁最小
    plan_id: int | None = None
    type_id: ScoreType = ScoreType.SkyLine


class Skyline(Algo):


    def itemSort_cmp(self, a: "Item", b: "Item"):
        chooseA, chooseB = 1, -1
        min_side_a = min(a.size.width, a.size.height)
        min_side_b = min(b.size.width, b.size.height)
        if min_side_a < min_side_b:
            return chooseB
        else:
            return chooseA

    def calc_waste_map_score(self, item: "Item", container: "Container") -> int | float:
        # GuillotineBinPack::RectBestShortSideFit:
        # 该规则试图找到一个空闲矩形，使得新矩形的一边（最短的那一边）和空闲矩形的相应边的长度差最小。
        if item.size.height>item.size.width:
            return container.rect.width - item.size.width
        else:
            return container.rect.height - item.size.height

    def compute_wasted_area(self, item: "Item", begin_idx, end_idx, containers: "list[Container]") -> tuple[int | float, int | float]:
        """

        :param containers:
        :param end_idx:
        :param begin_idx:
        :param item:
        :return: score,container
        """
        waste_area = 0

        height = item.size.height
        width = item.size.width
        start_x = containers[begin_idx].rect.start.x
        end_x = start_x + width
        min_y = containers[begin_idx].rect.start.y
        if begin_idx == end_idx:
            return waste_area, min_y
        for i in range(begin_idx, end_idx, 1):
            waste_area += (min_y - containers[i + 1].rect.start.y) * containers[i + 1].rect.width
        return waste_area, min_y
        pass

    def get_placable_area(self, item: "Item", begin_idx, containers: "list[Container]") -> float | int:
        """
        可放置区域满足:
            1 后继所有container的y值小于等于第一个container的y值
            2
        :param item:
        :param begin_idx:
        :param containers:
        :return:
            若有可放置的区域,返回end_idx,否则返回-1
        """
        debug_plan = []
        height = item.size.height
        width = item.size.width
        item_start_y = containers[begin_idx].rect.start.y
        item_start_x = containers[begin_idx].rect.start.x
        item_end_x = item_start_x + width
        item_end_y = item_start_y + height
        container_start_y = containers[begin_idx].rect.start.y
        container_end_y = containers[begin_idx].rect.end.y
        end_idx = begin_idx
        if item_end_x <= containers[end_idx].rect.end.x and container_start_y<=item_start_y and item_end_y <= container_end_y:
            return end_idx
        else:

            if end_idx+1==len(containers):
                return -1
            # 先判断占据的容器个数,再判断容器的高度是否满足条件
            for idx in range(begin_idx+1,len(containers)):
                if item_end_x <= containers[idx].rect.end.x and containers[idx].rect.end.y>=item_end_y and item_start_y>=containers[idx].rect.start.y:
                    end_idx=idx
                    break
            if end_idx==begin_idx:
                return -1
            # if item_end_x > containers[end_idx].rect.end.x:
            for idx in range(begin_idx, end_idx):
                if not (containers[idx].rect.end.y >= item_end_y and item_start_y >= containers[idx].rect.start.y):
                    return -1
            # for idx in range(begin_idx,end_idx):
            #     if containers[idx].rect.end.y>=item_end_y and item_start_y>=containers[idx].rect.start.y:
            #         continue
            #     else:
            #         return -1
            return end_idx

    def best_score_cmp(self, a: "ItemScore", b: "ItemScore"):
        chooseA, chooseB = 1, -1
        if a.type_id!=b.type_id:
            raise ValueError("比较类型要相同:",a,b)
        if a.plan_id == b.plan_id:
            if a.type_id == ScoreType.WasteMap and b.type_id == ScoreType.WasteMap:
                if a.score > b.score:
                    return chooseB
                else:
                    return chooseA
            elif a.type_id == ScoreType.WasteMap and b.type_id == ScoreType.SkyLine:
                return chooseA
            elif a.type_id == ScoreType.SkyLine and b.type_id == ScoreType.WasteMap:
                return chooseB
            else:
                # 此时是比较浪费空间,相等的时候比较海拔高度,都是越低越好
                if a.score[0] == b.score[0]:
                    if a.score[1] > b.score[1]:
                        return chooseB
                    else:
                        return chooseA
                else:
                    if a.score[0] > b.score[0]:
                        return chooseB
                    else:
                        return chooseA
        else:
            if a.plan_id > b.plan_id:
                return chooseB
            else:
                return chooseA
    def run(self, debug_mode=False):
        items = sorted(self.items, key=cmp_to_key(self.itemSort_cmp), reverse=True)

        plans: "list[Plan]" = []
        for new_item in items:
            scores: "list[ItemScore]" = []
            for plan in plans:
                # 先检查wastedMap,如果有,就不考虑天际线了
                for i in range(len(plan.wasteMap)):
                    # if self.is_waste_map_placable(new_item, plan.wasteMap[i]):
                    waste_rect = plan.wasteMap[i].rect
                    if new_item.size + waste_rect.start in waste_rect:
                        item = new_item.copy()
                        item.pos = waste_rect.start
                        scores.append(
                                ItemScore(
                                        item=item,
                                        container_range=(i, i + 1),
                                        score=self.calc_waste_map_score(item, plan.wasteMap[i]),
                                        plan_id=plan.ID,
                                        type_id=ScoreType.WasteMap
                                )
                        )
                    itemT = new_item.transpose()
                    if itemT.size + waste_rect.start in waste_rect:
                        item = itemT.copy()
                        item.pos = waste_rect.start
                        scores.append(
                                ItemScore(
                                        item=item,
                                        container_range=(i, i + 1),
                                        score=self.calc_waste_map_score(item, plan.wasteMap[i]),
                                        plan_id=plan.ID,
                                        type_id=ScoreType.WasteMap
                                )
                        )
            if len(scores) == 0:
                for plan in plans:
                    for i in range(len(plan.skyLineContainers)):
                        idx = self.get_placable_area(new_item, i, plan.skyLineContainers)
                        if idx >= i:
                            item = new_item.copy()
                            item.pos = plan.skyLineContainers[i].rect.start
                            scores.append(ItemScore(
                                    item=item,
                                    container_range=(i, idx + 1),
                                    score=self.compute_wasted_area(item, i, idx, plan.skyLineContainers),
                                    plan_id=plan.ID
                            ))

                        itemT = new_item.transpose()
                        idx = self.get_placable_area(itemT, i, plan.skyLineContainers)
                        if idx >= i:
                            item = itemT.copy()
                            item.pos = plan.skyLineContainers[i].rect.start
                            scores.append(ItemScore(
                                    item=item,
                                    container_range=(i, idx + 1),
                                    score=self.compute_wasted_area(item, i, idx, plan.skyLineContainers),
                                    plan_id=plan.ID
                            ))
            if len(scores) == 0:
                idx = self.get_placable_area(new_item, 0, [Container(self.material.start, self.material.end)])
                if idx >= 0:
                    item = new_item.copy()
                    scores.append(ItemScore(
                            item=item,
                            container_range=(0,1),
                            score=self.compute_wasted_area(item, 0, idx, [Container(self.material.start, self.material.end)]),
                            plan_id=-1
                    ))

                itemT = new_item.transpose()
                idx = self.get_placable_area(itemT, 0, [Container(self.material.start, self.material.end)])
                if idx >= 0:
                    item = itemT.copy()
                    scores.append(ItemScore(
                            item=item,
                            container_range=(0,1),
                            score=self.compute_wasted_area(item, 0, idx, [Container(self.material.start, self.material.end)]),
                            plan_id=-1
                    ))
            if len(scores)==0:
                raise Exception("没有找到合适的位置",new_item)
            best_score: "ItemScore" = min(scores, key=cmp_to_key(self.best_score_cmp))
            if best_score.plan_id == -1:
                plan = Plan(len(plans), self.material.copy(), [], [Container(self.material.start, self.material.end)], [])
                container_top = Container(best_score.item.size.topLeft, POS(
                        best_score.item.size.width,
                        self.material.topRight.y),
                                          plan.ID)
                container_right = Container(best_score.item.size.bottomRight, self.material.topRight)
                if container_right.rect == Rect:
                    plan.skyLineContainers.append(container_right)
                if container_top.rect == Rect:
                    plan.skyLineContainers.append(container_top)
                plans.append(plan)
                plan.item_sequence.append(best_score.item)
                if debug_mode:
                    standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加矩形,{best_score.item.rect}")
                # if debug_mode:
                #     standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, )
            else:
                plan = plans[best_score.plan_id]
                plan.item_sequence.append(best_score.item)
                if debug_mode:
                    standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加矩形,{best_score.item.rect}")

                new_rect = best_score.item.size + best_score.item.pos
                if best_score.type_id == ScoreType.WasteMap:  # wasteMap模式
                    # 将所占用的wastemap容器删除,填入新的wastemap容器
                    container = plan.wasteMap[best_score.container_range[0]]
                    plan.wasteMap.remove(container)
                    if debug_mode:
                        standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"移除wastemap容器,{container.rect}")


                    # GuillotineBinPack::SplitMaximizeArea:
                    # 这个规则会选择一种分割方式，使得分割后剩余的空闲矩形的面积最大
                    split_1 = (Container(new_rect.topLeft, container.rect.topRight, best_score.plan_id),
                               Container(new_rect.bottomRight, POS(container.rect.bottomRight.x, new_rect.topRight.y), best_score.plan_id)
                               )
                    split_2 = (Container(new_rect.topLeft, POS(new_rect.topRight.x, container.rect.topRight.y), best_score.plan_id),
                               Container(new_rect.bottomRight, container.rect.topRight, best_score.plan_id)
                               )
                    split_1_area = max([split_1[0].rect.area, split_1[1].rect.area])
                    split_2_area = max([split_2[0].rect.area, split_2[1].rect.area])
                    if split_1_area > split_2_area:
                        newC_top, newC_right = split_1
                    else:
                        newC_top, newC_right = split_2
                    if newC_top != Rect:
                        newC_top=None
                    if newC_right != Rect:
                        newC_right=None

                    # 判断新切出来的container是否能合并到旧的
                    for waste_c in plan.wasteMap:

                        if newC_top is None and newC_right is None:
                            break
                        if newC_right is not None:
                            result = newC_right.rect & waste_c.rect
                            if result == Line:
                                diff = result.end - result.start
                                if diff.x==0: # 竖轴为线
                                    if waste_c.rect.bottomRight == newC_right.rect.bottomLeft and\
                                            waste_c.rect.topRight==newC_right.rect.topLeft: #  新容器在右边
                                        waste_c.rect.end = newC_right.rect.end
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_right = None
                                    elif waste_c.rect.bottomLeft==newC_right.rect.bottomRight and waste_c.rect.topLeft ==newC_right.rect.topRight: # 新容器在左边
                                        waste_c.rect.start = newC_right.rect.start
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_right = None
                                elif diff.y==0: # 横轴线
                                    if waste_c.rect.topRight==newC_right.rect.bottomRight and waste_c.rect.topLeft==newC_right.rect.bottomLeft: # 新容器在上边
                                        waste_c.rect.end = newC_right.rect.end
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_right = None
                                    elif waste_c.rect.bottomRight==newC_right.rect.topRight and waste_c.rect.bottomLeft==newC_right.rect.topLeft: # 新容器在下边
                                        waste_c.rect.start = newC_right.rect.start
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_right = None


                        if newC_top is not None:
                            result = newC_top.rect & waste_c.rect
                            if result == Line:
                                diff = result.end - result.start
                                if diff.x == 0:  # 竖轴为线
                                    if waste_c.rect.bottomRight == newC_top.rect.bottomLeft and  waste_c.rect.topRight == newC_top.rect.topLeft:  # 新容器在右边
                                        waste_c.rect.end = newC_top.rect.end
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_top = None
                                    elif waste_c.rect.bottomLeft == newC_top.rect.bottomRight and waste_c.rect.topLeft == newC_top.rect.topRight:  # 新容器在左边
                                        waste_c.rect.start = newC_top.rect.start
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_top = None
                                elif diff.y == 0:  # 横轴线
                                    if waste_c.rect.topRight == newC_top.rect.bottomRight and waste_c.rect.topLeft == newC_top.rect.bottomLeft:  # 新容器在上边
                                        waste_c.rect.end = newC_top.rect.end
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_top = None
                                    elif waste_c.rect.bottomRight == newC_top.rect.topRight and waste_c.rect.bottomLeft == newC_top.rect.topLeft:  # 新容器在下边
                                        waste_c.rect.start = newC_top.rect.start
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接wastemap容器,{waste_c.rect}")
                                        newC_top = None
                                # if newC_top is None:


                    if newC_top is not None:
                        plan.wasteMap.append(newC_top)
                        if debug_mode:
                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加wastemap容器,{newC_top.rect}")
                    if newC_right is not None:
                        plan.wasteMap.append(newC_right)
                        if debug_mode:
                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加wastemap容器,{newC_right.rect}")
                    if debug_mode:
                        standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"最终效果")
                    pass
                else:  # skyline模式
                    # 根据score给的range取出全部横跨的skyline容器,并从freeContainers中删除
                    removed_containers = plan.skyLineContainers[best_score.container_range[0]:best_score.container_range[1]]
                    for container in removed_containers:
                        plan.skyLineContainers.remove(container)
                        if debug_mode:
                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"移除skyline容器,{container.rect}")
                    last_c = removed_containers[-1]
                    # 创建新的skyline,通常只有两个
                    container_top = Container(new_rect.topLeft, POS(
                            new_rect.topRight.x, self.material.height
                    ), plan.ID)

                    if new_rect.bottomRight.x!=last_c.rect.end.x:
                        container_right = Container(
                                POS(new_rect.bottomRight.x, last_c.rect.start.y),
                                POS(last_c.rect.end.x, self.material.height)
                        )
                    else:
                        container_right = None


                    # 合并到原有的skyline
                    for sky_c in plan.skyLineContainers:
                        if container_right is None and container_top is None:
                            break
                        if container_right is not None:
                            result = container_right.rect & sky_c.rect
                            if result == Line:
                                diff = result.end - result.start
                                if diff.x == 0:  # 表明此时是竖轴重合
                                    if sky_c.rect.bottomLeft == container_right.rect.bottomRight:  # 旧容器在右侧
                                        sky_c.rect.start = container_right.rect.start
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接skyline容器,{sky_c.rect}")
                                        container_right = None
                                    elif sky_c.rect.bottomRight == container_right.rect.bottomLeft:  # 旧容器在左侧
                                        sky_c.rect.end = container_right.rect.end
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接skyline容器,{sky_c.rect}")
                                        container_right = None
                        if container_top is not None:
                            result = container_top.rect & sky_c.rect
                            if result == Line:
                                diff = result.end - result.start
                                if diff.x == 0:
                                    if sky_c.rect.bottomLeft == container_top.rect.bottomRight:  # 旧容器在右侧
                                        sky_c.rect.start = container_top.rect.start
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接skyline容器,{sky_c.rect}")
                                        container_top = None
                                    elif sky_c.rect.bottomRight == container_top.rect.bottomLeft:  # 旧容器在左侧
                                        sky_c.rect.end = container_top.rect.end
                                        if debug_mode:
                                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"拼接skyline容器,{sky_c.rect}")
                                        container_top = None

                    if container_right is not None:
                        plan.skyLineContainers.append(container_right)
                        if debug_mode:
                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加skyline容器,{container_right.rect}")
                    if container_top is not None:
                        plan.skyLineContainers.append(container_top)
                        if debug_mode:
                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加skyline容器,{container_top.rect}")

                    # 同时也要计算浪费掉的矩形
                    waste_rect_to_append = []
                    last_waste = None
                    for waste_c in removed_containers[1:]:
                        if new_rect.bottomRight.y > waste_c.rect.bottomRight.y:
                            c = Container(waste_c.rect.bottomLeft, POS(
                                    min(waste_c.rect.bottomRight.x, new_rect.bottomRight.x),
                                    new_rect.bottomRight.y
                            ))
                            waste_rect_to_append.append(c)
                            # last_waste=c

                    for waste_c in waste_rect_to_append:
                        if waste_c is not None:
                            plan.wasteMap.append(waste_c)
                            if debug_mode:
                                standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加wastemap容器,{waste_c.rect}")

                    # 对剩余的skyline进行添加
                    if debug_mode:
                        standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"最终效果")

                    pass

                pass
            plan.skyLineContainers.sort(key=lambda x: x.rect.start.x)
            plan.wasteMap.sort(key=lambda x: x.rect.start.x)
        self.solution = plans

    pass


if __name__ == "__main__":
    np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(随机_data.shape[0], 500)
    data = 随机_data[data_idx]
    s = Skyline(data)

    # idx = s.get_placable_area(Item(0,Rect(0,0,1500,1500),POS(0,0)),0,[Container(POS(0,0),POS(*MATERIAL_SIZE))])
    # print(idx)
    print(s.task_id)
    # s.run(debug_mode=True)
    s.run(debug_mode=False)
    print(f"util rate={s.avg_util_rate()}")
    standard_draw_plan(s.solution,task_id=s.task_id)
    pass
