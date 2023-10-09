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
import numpy as np
from constant import *
from functools import cmp_to_key
from enum import Enum, auto


# 感知机模型和训练
# 训练后保存参数,使用时读取参数
class ScoreParameter:
    class V:
        SLP = "SLP"
        MLP6 = "MLP6"
        MLP4 = "MLP4"
    def __init__(self):
        self.version=self.V.MLP6



    def traning(self):
        pass

    def sort_parameter(self,item_width,item_height,item_maxL,item_minL,material_width,material_height)->float|int:
        pass

    def score_parameter(self,item_width,item_height,container_begin_x,container_begin_y,container_width,container_height,material_width,material_height,plan_id,plan_util_rate):



@dataclass
class ItemScore:
    item: Item
    container_range: tuple[int,int]
    score:tuple[ int | float,int|float]|int|float # 先比较浪费空间谁最小,然后比较min_y谁最小
    plan_id: int | None = None

class HistoryNet:
    def __init__(self,std_materials:list, std_items: np.ndarray, score_parameters:"ScoreParameter|None"=None
         ):
        """
        :param score_calculators:[level_score,corner_score,basic_score]
        :param edge_loss: [LEFT,BOTTOM,RIGHT,TOP]
        :param cutting_gap: 切割的缝隙
        :param std_materials:
            np.ndarray: [原始行号,较长,较短,长,宽,高,余量],
        :param std_items:
            np.ndarray: [原始行号,较长,较短,长,宽,高,余量,有纹路],
        :param  score_parameters:[新板,旧板,层次,角数,几个放置参数],
        :return:
        plan: [material_id,(((x1,y1),(x2,y2)),item_id),(((x1,y1),(x2,y2)),item_id)...],
        corners:[material_id,((x1,y1),(x2,y2)),((x1,y1),(x2,y2)),((x1,y1),(x2,y2)),...]
        """
        self.materials :"Rect" = Rect(POS(0, 0), POS(std_materials[COL.maxL], std_materials[COL.minL]))
        self.items = std_items
        self.score_parameters:"ScoreParameter" = score_parameters if score_parameters else ScoreParameter()


    def run(self):
        std_items = self.items
        std_materials = self.materials

        min_item_length = std_items[:, COL.minL].min()

        plan_updated_for_debug = []

        std_items = self.sorted_items(std_materials, std_items)

        plan_for_cutting: "list[list[float|int|tuple]]" = []  # 最终用于下料的方案 [material_id,(((x1,y1),(x2,y2)),item_id),(((x1,y1),(x2,y2)),item_id)...]
        corners_in_planed_materials: "list[list[float|int|tuple]]" = []  # 每个材料可以放置的位置, [material_id,((x1,y1),(x2,y2)),((x1,y1),(x2,y2)),((x1,y1),(x2,y2)),...]
        total_item_area = 0
        plan_material_area = 0
        LEFT, BOTTOM, RIGHT, TOP,cutting_gap = 0
        total_count = 0
        x, y = 0, 1
        for at_i in range(std_items.shape[0]):
            item = std_items[at_i]
            # [score, new_material_id, planed_material_idx, item_id, ((x1,y1),(x2,y2)), corner]
            i_maxL, i_minL, item_id, has_texture, item_remain = [float(i) for i in [item[COL.maxL], item[COL.minL], item[COL.ID], item[COL.Item.Texture], item[COL.Remain]]]
            item_remain = int(item_remain)
            remain_maxL = np.max(std_items[:, COL.maxL])

            # print("current item count = ",item_remain)
            for at_copy_j in range(item_remain):
                pre_score_list = []
                # new_material_candidates =
                # print("new_material_candidates=",new_material_candidates)
                # 处理旧板材
                for idx in range(len(corners_in_planed_materials)):
                    corners = corners_in_planed_materials[idx]
                    material_id = corners[0]
                    X, Y = self.materials  # std_materials[std_materials[:,COL.Material.ID]==material_id][0,:COL.minL+1]
                    legal_size = X - RIGHT, Y - TOP
                    for corner in corners[1:]:
                        (corner_start, corner_end) = corner
                        # m_maxL, m_minL = x2-del_corner_x0,y2-del_corner_y0
                        itemPos_end = (corner_start[x] + i_maxL, corner_start[y] + i_minL)
                        if mainFun_is_corner_acceptable(*itemPos_end, *corner_end, legal_X=legal_size[0], legal_Y=legal_size[1], cutting_gap=cutting_gap):
                            pre_score_list.append([0, -1, idx, item_id,
                                                   (corner_start, itemPos_end),
                                                   corner])
                        itemPos_end = (corner_start[x] + i_minL, corner_start[y] + i_maxL)
                        if mainFun_is_corner_acceptable(*itemPos_end, *corner_end, legal_X=legal_size[0], legal_Y=legal_size[1], cutting_gap=cutting_gap, has_texture=has_texture):
                            pre_score_list.append([0, -1, idx, item_id,
                                                   (corner_start, itemPos_end),  # 此处填写零件本身的尺寸
                                                   corner])  # 填写角的尺寸
                    if len(pre_score_list) > 0:
                        break

                if len(pre_score_list) == 0:
                    # 处理新板材
                    for material in new_material_candidates:
                        m_maxL, m_minL, material_id = [float(i) for i in [material[COL.maxL], material[COL.minL], material[COL.ID]]]
                        legal_size = m_maxL - RIGHT, m_minL - TOP
                        start_pos = LEFT + 0, BOTTOM + 0

                        if mainFun_is_corner_acceptable(i_maxL, i_minL, *legal_size, legal_X=legal_size[0], legal_Y=legal_size[1], cutting_gap=cutting_gap):
                            pre_score_list.append([0, material_id, -1, item_id,
                                                   (start_pos, (start_pos[0] + i_maxL, start_pos[1] + i_minL)),
                                                   (start_pos, legal_size)])
                        if mainFun_is_corner_acceptable(i_minL, i_maxL, *legal_size, legal_X=legal_size[0], legal_Y=legal_size[1], cutting_gap=cutting_gap, has_texture=has_texture):
                            pre_score_list.append([0, material_id, -1, item_id,
                                                   (start_pos, (start_pos[0] + i_minL, start_pos[1] + i_maxL)),
                                                   (start_pos, legal_size)])

                # 评分,获取方案
                if pre_score_list:

                    min_gap = min_item_length + cutting_gap
                    max_score = mainFun_get_best_score_plan(pre_score_list, plan_for_cutting, cutting_gap, std_materials, score_parameters, score_calculators)
                    score, new_material_id, plan_id, item_id, (itemPos_start, itemPos_end), (delCorner_start, delCorner_end) = max_score

                    cut_pos = itemPos_end[x] + cutting_gap, \
                              itemPos_end[y] + cutting_gap
                    new_BR_corner = (cut_pos[x], delCorner_start[y]), (delCorner_end[x], cut_pos[y])
                    new_top_corner = (delCorner_start[x], cut_pos[y]), delCorner_end

                    if new_BR_corner[1][x] - new_BR_corner[0][x] < min_gap:
                        new_BR_corner = None
                    if new_top_corner[1][y] - new_top_corner[0][y] < min_gap:
                        if new_BR_corner is not None:
                            new_BR_corner = new_BR_corner[0], new_top_corner[1]
                            new_top_corner = None

                    if new_material_id >= 0:  # 新建
                        # print("add to new plan")
                        idx = np.where(std_materials[:, COL.ID] == new_material_id)[0][0]
                        std_materials[idx][COL.Remain] -= 1
                        X, Y = get_material_XY(new_material_id, std_materials)
                        X, Y = float(X), float(Y)
                        legal_zone = X - RIGHT, Y - TOP
                        new_corner_plan = [new_material_id]

                        if new_BR_corner is not None:

                            if new_BR_corner[0][y] >= Y or new_BR_corner[0][x] >= X:
                                new_BR_corner = None
                            else:
                                new_corner_plan.append(new_BR_corner)
                        if new_top_corner is not None:
                            if new_top_corner[0][y] >= Y or new_top_corner[0][x] >= X:
                                new_top_corner = None
                            else:
                                new_corner_plan.append(new_top_corner)
                        corners_in_planed_materials.append(new_corner_plan)

                        # print("get new corner in planed material=",corners_in_planed_materials)
                        plan_for_cutting.append([new_material_id, ((itemPos_start, itemPos_end), item_id)])
                        plan_material_area += X * Y / 1000000
                        plan_updated_for_debug.append(len(plan_for_cutting) - 1)

                    elif plan_id >= 0:  # 原有
                        plan = plan_for_cutting[plan_id]
                        plan.append(((itemPos_start, itemPos_end), item_id))
                        corners = corners_in_planed_materials[plan_id]
                        corners.remove((delCorner_start, delCorner_end))

                        for corner_at_i in range(len(corners)):
                            if corner_at_i == 0: continue
                            old_corner = corners[corner_at_i]
                            if new_BR_corner is None and new_top_corner is None:
                                break
                            if new_BR_corner is not None:
                                if new_BR_corner[0] == (old_corner[0][x], old_corner[1][y]) and new_BR_corner[1][x] == old_corner[1][x] and new_BR_corner[1][y] >= old_corner[1][y]:
                                    corners[corner_at_i] = old_corner[0], new_BR_corner[1]
                                    new_BR_corner = None
                            if new_top_corner is not None:
                                if (new_top_corner[0][x], new_top_corner[1][y]) == old_corner[0] and new_top_corner[1][x] == old_corner[1][x] and new_top_corner[1][y] <= old_corner[1][y]:
                                    corners[corner_at_i] = old_corner[0], new_top_corner[1]
                                    new_top_corner = None

                        if new_top_corner is not None and new_top_corner[1][y] - new_top_corner[0][y] >= min_gap and new_top_corner[1][x] - new_top_corner[0][x] >= min_gap:
                            corners.append(new_top_corner)
                        if new_BR_corner is not None and new_BR_corner[1][x] - new_BR_corner[0][x] >= min_gap and new_BR_corner[1][y] - new_BR_corner[0][y] >= min_gap:
                            corners.append(new_BR_corner)

                        plan_updated_for_debug.append(plan_id)

                    else:
                        raise ValueError("存在错误请检查")


                else:
                    if handle_emptyscore_error == ErrorHandle.raise_error:
                        item_info = {
                                "长"  : item[COL.length],
                                "宽"  : item[COL.width],
                                "总量": item[COL.Remain],
                        }
                        material_info = [{
                                "长"  : materials[COL.length],
                                "宽"  : materials[COL.width],
                                "余量": materials[COL.Remain],
                        }
                                for materials in std_materials
                        ]
                        if material_info[0]["余量"] == 0:
                            raise ValueError("计算错误,板材数量不足")
                        else:
                            raise ValueError("计算错误,板材尺寸不足")

                    elif handle_emptyscore_error == ErrorHandle.return_empty:
                        warnings.warn("请检查配置,当前item没有合适的装载方案")
                        return [], []

                if need_report:
                    total_count += 1
                    print("\rrate=", total_item_area / plan_material_area, "plan count=", len(plan_for_cutting), "item count=", total_count, end="", flush=True)
                if need_step_img:
                    for plan_id in set(plan_updated_for_debug):
                        material_id = plan_for_cutting[plan_id][0]
                        X, Y = get_material_XY(material_id, std_materials)
                        debug_draw(plan_id, plan_for_cutting, (X, Y), corners_in_planed_materials, f"{at_i},{at_copy_j}")

                plan_updated_for_debug = []
        if need_report:
            print("")
        return plan_for_cutting, corners_in_planed_materials


    def sorted_items(self,std_materials, std_items):
        """
            排序长宽比,再按较长/最长边排序,按较短/最短边排序
            :param items:  np.ndarray: [原始行号,较长,较短,长,宽,高,余量,有纹路]
            :return:np.ndarray
            """
        # 计算较长边和较短边的最大值
        data = std_items
        maxL = np.max(data[:, 1])  # 假设较长边在第二列
        maxl = np.max(data[:, 2])  # 假设较短边

        # 在第三列
        max_area = maxL * maxl
        # # 计算每一行的权重
        # weights = data[:, 2] / data[:, 1]*score_parameters[0] + data[:, 1] / maxL*score_parameters[1]+data[:, 2] / maxl*score_parameters[2]

        weights = self.score_parameters.sort_parameter(

        )
        # weights = data[:, 2]/maxl+10*data[:, 1]/maxL
        # 使用argsort函数得到排序后的索引
        sorted_indices = np.argsort(weights)[::-1]  # [::-1]用于实现降序排序

        # 使用这些索引对原始数组进行排序
        sorted_data = data[sorted_indices]
        return sorted_data