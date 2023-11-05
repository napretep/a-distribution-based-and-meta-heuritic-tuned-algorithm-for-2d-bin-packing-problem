# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'max_rect.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 17:43'
"""
import time
import uuid
from dataclasses import dataclass
from functools import cmp_to_key
from constant import *
import numpy as np
from visualizing.draw_plan import standard_draw_plan




@dataclass
class Plan(ProtoPlan):
    freeContainers: "list[Container]|None"=None

    def get_remain_containers(self):
        return self.freeContainers

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
    assert a.score[1]>=0 and a.score[0]>=0
    assert b.score[1]>=0 and b.score[0]>=0
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


class MaxRect(Algo):

    # def __init__(self, item_data: "np.ndarray", material_data: "list",task_id=None):
    #     """
    #     :param item_data: [ID,maxL,minL]
    #     :param material_data: [ID,maxL,minL]
    #     :return:
    #     """
    #     super(item_data, material_data, task_id)
    #
    def run(self,debug_mode=False):
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
        debug_plan:"list[Plan]"=[]
        plans: "list[Plan]" = []
        # containers:"list[list[Container]]" = [] # container的id对应了plan
        while len(self.items) > 0:

            scores: "list[ItemScore]" = []
            # if debug_mode:
            #     for plan in plans:
            #         print(f"plan.freeContainers count={len(plan.freeContainers)}")


            for new_item in self.items:
                for plan in plans:
                    for container in plan.freeContainers:

                        if (new_item.size + container.rect.start) in container.rect:
                            new_item.pos = container.rect.start
                            score = ItemScore(
                                    item=new_item.copy(),
                                    container=container,
                                    plan_id=plan.ID,
                                    score=(container.rect.width - new_item.size.width, container.rect.height - new_item.size.height)
                            )
                            # if debug_mode:
                            #     print(min(score.score), end=",")
                            scores.append(score)
                        itemT: "Item" = new_item.transpose()
                        itemT.pos=POS()
                        if (itemT.size + container.rect.start) in container.rect:
                            itemT.pos = container.rect.start
                            score = ItemScore(
                                    item=itemT.copy(),
                                    container=container,
                                    plan_id=plan.ID,
                                    score=(container.rect.width - itemT.size.width, container.rect.height - itemT.size.height)
                            )
                            # if debug_mode:
                            #     print(min(score.score), end=",")
                            scores.append(score)
                new_item.pos = POS()
                new_container=Container(POS(0, 0), POS(self.material.width, self.material.height))

                if (new_item.size+new_container.rect.start) in new_container.rect:
                    score = ItemScore(
                            item=new_item.copy(),
                            container=new_container,
                            plan_id=-1,
                            score=(self.material.width - new_item.size.width, self.material.height - new_item.size.height)
                    )
                    # if debug_mode:
                    #     print(min(score.score), end=",")
                    scores.append(score)
                itemT: "Item" = new_item.transpose()
                itemT.pos=POS()

                if (itemT.size + new_container.rect.start) in new_container.rect:
                    score = ItemScore(
                            item=itemT.copy(),
                            container=new_container,
                            plan_id=-1,
                            score=(self.material.width - itemT.size.width, self.material.height - itemT.size.height)
                    )
                    # if debug_mode:
                    #     print(min(score.score), end=",")
                    scores.append(score)
            if len(scores)==0:
                raise ValueError(f"scores 空", f"当前itemlist = {self.items}")
            # if debug_mode:
            #     print(f"scores size ={len(scores)},item count = {len(self.items)}")
            best_score = min(scores,key=lambda x:min(x.score))
            # if debug_mode:
            #     print(f"best_score:{min(best_score.score)},best_item={best_score.item.rect},best_id={best_score.plan_id},best_place={best_score.container.rect}")
            if best_score.plan_id == -1:
                plan = Plan(ID=len(plans),material=self.material.copy(),item_sequence=[],freeContainers=[])
                item_rect = best_score.item.size+best_score.item.pos
                container_top = Container(item_rect.topLeft, self.material.topRight)
                container_btm = Container(item_rect.bottomRight, self.material.topRight)
                plan.freeContainers += [container_top, container_btm]
                plan.item_sequence.append(best_score.item)
                plans.append(plan)
                if debug_mode:
                    plan.remain_containers=plan.freeContainers
                    debug_plan.append(plan)
                if debug_mode:
                    standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"初始化添加容器")
            else:
                plan = plans[best_score.plan_id]
                plan.item_sequence.append(best_score.item)
                new_item = best_score.item
                container = best_score.container
                new_rect = new_item.size + new_item.pos
                if debug_mode:
                    standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加矩形,{new_rect}")
                container_to_remove=[]
                container_to_append=[]
                # 首先找到与插入新矩形有交集的容器,并删除他们,并根据这些容器创建新的不交新矩形的容器
                for free_c in plan.freeContainers:
                    result = free_c.rect & new_rect
                    # if debug_mode:
                    #     print("free_c=",free_c)
                    # 判断新加入的矩形是否和其他空余矩形有交集,如果有,则剔除这个矩形,同时生成它的剩余部分作为新的空余矩形
                    if result == Rect:
                        container_to_remove.append(free_c)
                        # 上部
                        if result.topRight.y < free_c.rect.topRight.y:
                            top_c = Container(
                                    POS(free_c.rect.topLeft.x, result.topLeft.y),
                                    free_c.rect.topRight,
                                    plan.ID
                            )
                            if top_c.rect == Rect:  # 还需要判断新容器是个有面积的矩形
                                container_to_append.append(top_c)
                                # if debug_mode:
                                #     print(f"container_to_append.append(top_c)={top_c}")
                        # 下部
                        if result.bottomRight.y > free_c.rect.bottomRight.y:
                            bottom_c = Container(free_c.rect.bottomLeft, POS(free_c.rect.bottomRight.x, result.bottomRight.y), plan.ID)
                            if bottom_c.rect == Rect:
                                container_to_append.append(bottom_c)
                                # if debug_mode:
                                #     print(f"container_to_append.append(bottom_c)={bottom_c}")
                        # 右部
                        if result.topRight.x < free_c.rect.topRight.x:
                            left_c = Container(POS(result.bottomRight.x, free_c.rect.bottomLeft.y), free_c.rect.topRight, plan.ID)
                            if left_c.rect == Rect:
                                container_to_append.append(left_c)
                                # if debug_mode:
                                #     print(f"container_to_append.append(left_c)={left_c}")
                        # 左部
                        if result.topLeft.x > free_c.rect.topLeft.x:
                            right_c = Container(free_c.rect.bottomLeft, POS(result.topLeft.x, free_c.rect.topRight.y), plan.ID)
                            if right_c.rect == Rect:
                                container_to_append.append(right_c)
                                # if debug_mode:
                                #     print(f"container_to_append.append(right_c)={right_c}")
                for container in container_to_remove:
                    if debug_mode:
                        standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"删除容器,{container.rect}")
                    plan.freeContainers.remove(container)
                    # if debug_mode:
                    #     print(f"plan.freeContainers.remove(container)={container}")

                container_to_remove = []
                # 维护矩形
                for free_c in plan.freeContainers:
                    for i in range(len(container_to_append)):
                        if container_to_append[i] is not None:
                            result = container_to_append[i].rect
                            # 清理被包含的container
                            if result & free_c.rect == result:  # 被包含
                                container_to_append[i] = None
                            # 合并公共边的container
                            elif result == Line:  # 相切则合并
                                diff = result.end - result.start
                                if diff.x == 0:
                                    if result.start == free_c.rect.bottomRight:
                                        free_c.rect.end = container_to_append[i].rect.end
                                    elif result.start == free_c.rect.bottomLeft:
                                        free_c.rect.start = container_to_append[i].rect.start
                                elif diff.y == 0:
                                    if result.end == free_c.rect.topRight:
                                        free_c.rect.end = container_to_append[i].rect.end
                                    elif result.end == free_c.rect.bottomRight:
                                        free_c.rect.start = container_to_append[i].rect.start
                                container_to_append[i] = None
                    if all(new_c is None for new_c in container_to_append):
                        break
                for container in container_to_append:
                    if container is not None and container not in plan.freeContainers:
                        if debug_mode:
                            standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加容器,{container.rect}")
                        plan.freeContainers.append(container)
                        # if debug_mode:
                        #     print(f"plan.freeContainers.append(container)={container}")

                if debug_mode:
                    plan.remain_containers=plan.freeContainers
                    debug_plan.append(plan)


            self.items.remove(best_score.item)
            if debug_mode:
                # print(len(self.items))
                if plans:
                    standard_draw_plan(debug_plan,is_debug=debug_mode,task_id=self.task_id)
                    debug_plan=[]
            # if debug_mode:
            #     print(plan.freeContainers)

        for plan in plans:
            plan.remain_containers=plan.freeContainers
        self.solution=plans

        return plans
    def clear_step_img(self):
        pass

    pass


if __name__ == "__main__":
    data_idx = np.random.choice(华为杯_data.shape[0],300)
    data = 华为杯_data[data_idx]
    r = MaxRect(data)
    print(r.task_id)
    plans = r.run(debug_mode=False)
    print(r.avg_util_rate())
    standard_draw_plan(r.solution,task_id=r.task_id)
