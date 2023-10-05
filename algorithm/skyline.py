# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'skyline.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 17:44'
"""
from dataclasses import dataclass

# - SKYLINE-MW-WM-BFF-DESCSS算法实现
# SKYLINE - 表示使用Skyline数据结构进行打包
# MW - 表示使用Skyline的最大适配(Best Fit)放置规则
# WM - 表示结合使用Waste Map技术优化
# BFF - 表示允许打包到多个箱子中(多箱打包)
# DESCSS - 表示按照矩形的短边长度降序排序。
import numpy as np
from constant import *
from functools import cmp_to_key


@dataclass
class Container:
    rect: Rect
    plan_id: int = None

    def __init__(self, start: POS, end: POS, plan_id: "int|None" = None):
        self.rect = Rect(start, end)
        self.plan_id = plan_id

    def __eq__(self, other: "Container"):
        return self.rect == other.rect
@dataclass
class Plan(ProtoPlan):
    freeContainers: list[Container]
    wasteMap: list[Container]

@dataclass
class ItemScore:
    item: Item
    container: Container
    score: int | float
    plan_id: int | None = None


class Skyline:

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

    def itemSort_cmp(self, a: "Item", b: "Item"):
        chooseA,chooseB = 1,-1
        min_side_a=min(a.size.width,a.size.height)
        min_side_b=min(b.size.width,b.size.height)
        if min_side_a<min_side_b:
            return chooseB
        else:
            return chooseA

    def score_calc(self,item:"Item",container:"Container")->int|float:


        return


        pass
    def run(self):
        items = sorted(self.items, key=cmp_to_key(self.itemSort_cmp))
        plans: "list[Plan]" = []
        for new_item in items:
            scores: "list[ItemScore]" = []
            for plan in plans:
                for container in plan.freeContainers:
                    if (new_item.size + container.rect.start) in container.rect:
                        new_item.pos = container.rect.start
                        score = ItemScore(
                                item=new_item,
                                container=container,
                                plan_id=plan.ID,
                                score=self.score_calc(new_item,container)
                        )
                        scores.append(score)
                        itemT: "Item" = new_item.transpose()
                        score = ItemScore(
                                item=itemT,
                                container=container,
                                plan_id=plan.ID,
                                score=self.score_calc(new_item,container)
                        )
                        scores.append(score)
                container = Container(POS(0, 0), POS(self.material.width, self.material.height))
                new_item.pos = POS()
                score = ItemScore(
                        item=new_item,
                        container=container,
                        plan_id=-1,
                        score=self.score_calc(new_item,container)
                )
                scores.append(score)
                itemT: "Item" = new_item.transpose()
                score = ItemScore(
                        item=itemT,
                        container=container,
                        plan_id=-1,
                        score=self.score_calc(new_item,container)
                )
                scores.append(score)
    pass
