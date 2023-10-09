# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'max_rect.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 17:43'
"""
from dataclasses import dataclass
from functools import cmp_to_key
from constant import *
import numpy as np


@dataclass
class Container:
    rect: Rect
    # intersect_with: "list[Container]"
    # containedBy: "list[Container]"
    plan_id: int = None

    def __init__(self, start: POS, end: POS, plan_id: "int|None" = None):
        self.rect = Rect(start, end)
        self.plan_id = plan_id

    def __eq__(self, other: "Container"):
        return self.rect == other.rect


@dataclass
class Plan(ProtoPlan):
    freeContainers: "list[Container]|None"=None


@dataclass
class ItemScore:
    item: Item
    container: Container
    score: tuple[int | float, int | float]
    plan_id: int | None = None


# - MAXRECTS-BSSF-BBF-GLOBAL算法实现
#     MAXRECTS:表示该算法使用了Maximal Rectangles的数据结构来维护空白区域
#     BSSF:表示使用了Best Short Side Fit的启发式规则,即选择剩余短边最小的空白区域放置物品。
#     BBF:表示使用了Bin Best Fit的多箱选择规则,即选择最佳箱子放置物品。
#     GLOBAL:表示使用了全局最优选择策略,即在所有未放置的物品中选择最佳的一个放置。


def score_cmp(a: "ItemScore", b: "ItemScore"):
    """
    :param a:
    :param b:
    :return:
    """
    chooseA, chooseB = 1, -1
    if a.score[1] == b.score[1]:
        if a.score[0] > b.score[0]:
            return chooseB
        else:
            return chooseA
    else:
        if a.score[1] > b.score[1]:
            return chooseB
        else:
            return chooseA


class MaxRect:

    def __init__(self, item_data: "np.ndarray", material_data: "np.ndarray"):
        """
        :param item_data: [ID,maxL,minL]
        :param material_data: [ID,maxL,minL]
        :return:
        """
        self.items: "list[Item]" = [Item(ID=item[0],
                                         size=Rect(POS(0, 0), POS(item[1], item[2])),
                                         pos=POS(0, 0)
                                         ) for item in item_data]
        self.material: "Rect" = Rect(POS(0, 0), POS(material_data[1], material_data[2]))
        self.solution: "np.ndarray|None" = None
        self.min_size = min(np.min(item_data[:, COL.minL]), np.min(item_data[:, COL.maxL]))

    def run(self):
        """
        循环 直到待排物品为空
            循环 取出待排物品
                创建评分表
                循环 取出方案的剩余容器(满足的)
                    求出物品在剩余容器中的平分,并插入评分表
                    旋转再评分一次
                在新材料的容器中放置物品进行评分,并插入评分表
            循环结束
            取出分数最高的结果,将物品更新到对应方案,剔除该物品
            根据插入的物品更新对应方案的剩余容器

        :return:
        """
        plans: "list[Plan]" = []
        # containers:"list[list[Container]]" = [] # container的id对应了plan
        while len(self.items) > 0:
            scores: "list[ItemScore]" = []
            for new_item in self.items:
                for plan in plans:
                    for container in plan.freeContainers:
                        if (new_item.size + container.rect.start) in container.rect:
                            new_item.pos = container.rect.start
                            score = ItemScore(
                                    item=new_item,
                                    container=container,
                                    plan_id=plan.ID,
                                    score=(container.rect.width - new_item.size.width, container.rect.height - new_item.size.height)
                            )
                            scores.append(score)
                        itemT: "Item" = new_item.transpose()
                        if (itemT.size + container.rect.start) in container.rect:
                            score = ItemScore(
                                    item=itemT,
                                    container=container,
                                    plan_id=plan.ID,
                                    score=(container.rect.width - itemT.size.width, container.rect.height - itemT.size.height)
                            )
                            scores.append(score)
                new_item.pos = POS()
                score = ItemScore(
                        item=new_item,
                        container=Container(POS(0, 0), POS(self.material.width, self.material.height)),
                        plan_id=-1,
                        score=(self.material.width - new_item.size.width, self.material.height - new_item.size.height)
                )
                scores.append(score)
                itemT: "Item" = new_item.transpose()
                score = ItemScore(
                        item=itemT,
                        container=Container(POS(0, 0), POS(self.material.width, self.material.height)),
                        plan_id=-1,
                        score=(self.material.width - itemT.size.width, self.material.height - itemT.size.height)
                )
                scores.append(score)
            best_score: "ItemScore" = min(scores, key=cmp_to_key(score_cmp))
            if best_score.plan_id == -1:
                plan = Plan(ID=len(plans),material=self.material.copy(),item_sequence=[],freeContainers=[])
                container_top = Container(best_score.item.size.topLeft, self.material.topRight)
                container_btm = Container(best_score.item.size.bottomRight, self.material.topRight)
                plan.freeContainers += [container_top, container_btm]
                plan.item_sequence.append(best_score.item)
            else:
                plan = plans[best_score.plan_id]
                # 取出旧container,item
                new_item = best_score.item
                container = best_score.container
                new_rect = new_item.size + new_item.pos
                # 由旧container和item,创建新container
                plan.freeContainers.remove(container)
                container_new1: "Container|None" = Container(container.rect.start, container.rect.end,plan.ID)
                container_new2: "Container|None" = Container(container.rect.start, container.rect.end,plan.ID)
                if container.rect.height > new_rect.height:
                    container_new1.rect.start = new_rect.topLeft
                else:
                    container_new1 = None
                if container.rect.width > new_rect.width:
                    container_new2.rect.start = new_rect.bottomRight
                else:
                    container_new2 = None
                if container_new1:
                    plan.freeContainers.append(container_new1)
                if container_new2:
                    plan.freeContainers.append(container_new2)

                # 更新container关系,根据item切割有交集的container
                wait_for_remove = []
                wait_for_append = []
                # top_c,bottom_c,left_c,right_c=None,None,None,None
                for free_c in plan.freeContainers:
                    result = free_c.rect & new_rect
                    # 判断新加入的矩形是否和其他空余矩形有交集,如果有,则剔除这个矩形,同时生成它的剩余部分作为新的空余矩形
                    if result == Rect:
                        wait_for_remove.append(free_c)
                        if result.topRight.y < free_c.rect.topRight.y:
                            top_c = Container(POS(free_c.rect.topLeft.x, result.topLeft.y), free_c.rect.topRight,plan.ID)
                            if top_c.rect==Rect:
                                wait_for_append.append(top_c)
                        if result.bottomRight.y > free_c.rect.bottomRight.y:
                            bottom_c = Container(free_c.rect.bottomLeft, POS(free_c.rect.bottomRight.x, result.bottomRight.y),plan.ID)
                            if bottom_c.rect == Rect:
                                wait_for_append.append(bottom_c)
                        if result.topRight.x < free_c.rect.topRight.x:
                            left_c = Container(POS(result.bottomRight.x, free_c.rect.bottomLeft.y), free_c.rect.topRight,plan.ID)
                            if left_c.rect == Rect:
                                wait_for_append.append(left_c)
                        if result.topLeft.x > free_c.rect.topLeft.x:
                            right_c = Container(free_c.rect.bottomLeft, POS(result.bottomRight.x, free_c.rect.topRight.y),plan.ID)
                            if right_c.rect==Rect:
                                wait_for_append.append(right_c)
                for container in wait_for_remove:
                    plan.freeContainers.remove(container)
                # 清理被包含的container
                # 合并公共边的container
                for free_c in plan.freeContainers:
                    for i in range(len(wait_for_append)):
                        if wait_for_append[i] is not None:
                            result = wait_for_append[i].rect
                            if result & free_c.rect == result:
                                wait_for_append[i] = None
                            elif result == Line:
                                diff = result.end - result.start
                                if diff.x == 0:
                                    if result.start == free_c.rect.bottomRight :
                                        free_c.rect.end = wait_for_append[i].rect.end
                                    elif result.start == free_c.rect.bottomLeft:
                                        free_c.rect.start = wait_for_append[i].rect.start
                                elif diff.y == 0:
                                    if result.end == free_c.rect.topRight:
                                        free_c.rect.end = wait_for_append[i].rect.end
                                    elif result.end == free_c.rect.bottomRight:
                                        free_c.rect.start = wait_for_append[i].rect.start
                                wait_for_append[i] = None
                    if all(new_c is None for new_c in wait_for_append):
                        break
                for container in wait_for_append:
                    if container is not None:
                        plan.freeContainers.append(container)
            self.items.remove(best_score.item)
        # return plans
        self.solution=plans
    def clear_step_img(self):
        pass

    pass


if __name__ == "__main__":

    MaxRect()
    pass