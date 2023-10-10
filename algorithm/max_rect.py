# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'max_rect.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 17:43'
"""
import uuid
from dataclasses import dataclass
from functools import cmp_to_key
from constant import *
import numpy as np
from visualizing.draw_plan import standard_draw_plan




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


class MaxRect:

    def __init__(self, item_data: "np.ndarray", material_data: "list",task_id=None):
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
        self.solution: "list[Plan]|None" = None
        self.min_size = min(np.min(item_data[:, COL.minL]), np.min(item_data[:, COL.maxL]))
        self.task_id = task_id if task_id else str(uuid.uuid4())[0:8]
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
                    scores.append(score)

            best_score = min(scores,key=lambda x:min(x.score))
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
                        # 下部
                        if result.bottomRight.y > free_c.rect.bottomRight.y:
                            bottom_c = Container(free_c.rect.bottomLeft, POS(free_c.rect.bottomRight.x, result.bottomRight.y), plan.ID)
                            if bottom_c.rect == Rect:
                                container_to_append.append(bottom_c)
                        # 右部
                        if result.topRight.x < free_c.rect.topRight.x:
                            left_c = Container(POS(result.bottomRight.x, free_c.rect.bottomLeft.y), free_c.rect.topRight, plan.ID)
                            if left_c.rect == Rect:
                                container_to_append.append(left_c)
                        # 左部
                        if result.topLeft.x > free_c.rect.topLeft.x:
                            right_c = Container(free_c.rect.bottomLeft, POS(result.topLeft.x, free_c.rect.topRight.y), plan.ID)
                            if right_c.rect == Rect:
                                container_to_append.append(right_c)

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
                for container in container_to_remove:
                    if debug_mode:
                        standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"删除容器,{container.rect}")
                    plan.freeContainers.remove(container)
                for container in container_to_append:
                    if debug_mode:
                        standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加容器,{container.rect}")
                    plan.freeContainers.append(container)
                # # 取出旧container,item
                # new_item = best_score.item
                # container = best_score.container
                # # print(new_item.pos)
                # new_rect = new_item.size + new_item.pos
                # # 由旧container和item,创建新container
                # if debug_mode:
                #     standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"删除容器,{container.rect}")
                # plan.freeContainers.remove(container)
                #
                #     # 将这些容器插入到表格中
                # container_new1: "Container|None" = Container(new_rect.topLeft, container.rect.end, plan.ID)
                # container_new2: "Container|None" = Container(new_rect.bottomRight, container.rect.end, plan.ID)
                # if container_new1.rect != Rect:
                #     container_new1 = None
                # if container_new2.rect != Rect:
                #     container_new2 = None
                #
                # for container in plan.freeContainers:
                #     if container_new1 is not None:
                #         if container.rect&container_new1.rect==container_new1.rect:
                #             container_new1=None
                #     if container_new2 is not None:
                #         if container.rect & container_new2.rect == container_new2.rect:
                #             container_new2=None
                #     if container_new1 is None and container_new2 is None:
                #         break
                #
                # wait_for_remove = []
                # wait_for_append = []
                # # 维护分两部分
                # # 第1部分: 原有的容器被新插入的矩形分割得到的两个新容器,如果他们中有完全被剩余容器包裹的, 删除掉
                # # 第2部分: 新插入的矩形分割了其他矩形,删除掉,添加没被分割的部分, 也要判断是否被其他容器包裹, 若是也要删除
                # # 最后,拼接可以合并的矩形
                #
                # # 首先找到与插入新矩形有交集的容器,并删除他们,并根据这些容器创建新的不交新矩形的容器
                # for free_c in plan.freeContainers:
                #
                #     result = free_c.rect & new_rect
                #     # 判断新加入的矩形是否和其他空余矩形有交集,如果有,则剔除这个矩形,同时生成它的剩余部分作为新的空余矩形
                #     # print(result)
                #     if result == Rect:
                #         # print("result == Rect","free_c=",free_c,"new_rect= ",new_rect)
                #         wait_for_remove.append(free_c)
                #
                #         # 上部
                #         if result.topRight.y < free_c.rect.topRight.y:
                #             top_c = Container(
                #                     POS(free_c.rect.topLeft.x, result.topLeft.y),
                #                     free_c.rect.topRight,
                #                     plan.ID
                #             )
                #             if top_c.rect == Rect:  # 还需要判断新容器是个有面积的矩形
                #                 wait_for_append.append(top_c)
                #         # 下部
                #         if result.bottomRight.y > free_c.rect.bottomRight.y:
                #             bottom_c = Container(free_c.rect.bottomLeft, POS(free_c.rect.bottomRight.x, result.bottomRight.y), plan.ID)
                #             if bottom_c.rect == Rect:
                #                 wait_for_append.append(bottom_c)
                #         # 右部
                #         if result.topRight.x < free_c.rect.topRight.x:
                #             left_c = Container(POS(result.bottomRight.x, free_c.rect.bottomLeft.y), free_c.rect.topRight, plan.ID)
                #             if left_c.rect == Rect:
                #                 wait_for_append.append(left_c)
                #         # 左部
                #         if result.topLeft.x > free_c.rect.topLeft.x:
                #             right_c = Container(free_c.rect.bottomLeft, POS(result.topLeft.x, free_c.rect.topRight.y), plan.ID)
                #             if right_c.rect == Rect:
                #                 wait_for_append.append(right_c)
                #
                #
                # for c in wait_for_append:
                #     if container_new1 is not None:
                #         if c.rect & container_new1.rect == container_new1.rect:
                #             container_new1=None
                #     if container_new2 is not None:
                #         if c.rect & container_new2.rect == container_new2.rect:
                #             container_new2=None
                #     if container_new1 is None and container_new2 is None:
                #         break
                #
                # if container_new1 is not None:
                #     wait_for_append.append(container_new1)
                # if container_new2 is not None:
                #     wait_for_append.append(container_new2)
                #
                # for c in wait_for_remove:
                #     if debug_mode:
                #         standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"删除容器,{c.rect}")
                #     plan.freeContainers.remove(c)
                #
                # # 维护矩形
                # for free_c in plan.freeContainers:
                #     for i in range(len(wait_for_append)):
                #         if wait_for_append[i] is not None:
                #             result = wait_for_append[i].rect
                #             # 清理被包含的container
                #             if result & free_c.rect == result:  # 被包含
                #                 wait_for_append[i] = None
                #             # 合并公共边的container
                #             elif result == Line:  # 相切则合并
                #                 diff = result.end - result.start
                #                 if diff.x == 0:
                #                     if result.start == free_c.rect.bottomRight:
                #                         free_c.rect.end = wait_for_append[i].rect.end
                #                     elif result.start == free_c.rect.bottomLeft:
                #                         free_c.rect.start = wait_for_append[i].rect.start
                #                 elif diff.y == 0:
                #                     if result.end == free_c.rect.topRight:
                #                         free_c.rect.end = wait_for_append[i].rect.end
                #                     elif result.end == free_c.rect.bottomRight:
                #                         free_c.rect.start = wait_for_append[i].rect.start
                #                 wait_for_append[i] = None
                #     if all(new_c is None for new_c in wait_for_append):
                #         break
                #
                # for container in wait_for_append:
                #     if container is not None:
                #         if debug_mode:
                #             standard_draw_plan([plan], is_debug=debug_mode, task_id=self.task_id, text=f"添加容器,{container.rect}")
                #         plan.freeContainers.append(container)


                # 下面开始第二部分

                if debug_mode:
                    plan.remain_containers=plan.freeContainers
                    debug_plan.append(plan)
            self.items.remove(best_score.item)
            if debug_mode:
                # print(len(self.items))
                if plans:
                    standard_draw_plan(debug_plan,is_debug=debug_mode,task_id=self.task_id)
                    debug_plan=[]
        # return plans
        print([i.util_rate() for i in plans])
        for plan in plans:
            plan.remain_containers=plan.freeContainers
        self.solution=plans

        return plans
    def clear_step_img(self):
        pass

    pass


if __name__ == "__main__":
    # r1 = Rect(50,50,100,100)
    # r2 = Rect(75,75,120,95)
    # r3 = Rect(25,75,75,95)
    # r4 = Rect(75,75,120,120)
    # r5 = Rect(60,10,80,60)
    # print(r1 & r2)
    # print(r1 & r5)


    data_idx = np.random.choice(华为杯_data.shape[0],300)
    data = 华为杯_data[data_idx]
    r = MaxRect(data,[0,2440,1220])
    print(r.task_id)
    plans = r.run(debug_mode=False)
    print(r.solution)
    standard_draw_plan(plans,task_id=r.task_id)
    pass