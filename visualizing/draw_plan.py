# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'draw_plan.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/3 18:19'
"""
import uuid

# -*- coding: utf-8 -*-
"""
__project_ = '矩形材料切割'
__file_name__ = 'visualizing.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/8/27 8:51'
"""
import os
import shutil
import sys, base64, io
import time
from datetime import datetime
import matplotlib.pyplot as plot
import matplotlib.patches as patches
from constant import Rect,POS, PROJECT_ROOT_PATH, PROCESSING_STEP_PATH, IMAGES_PATH,SOLUTIONS_PATH,ProtoPlan



def _draw_text(plot, p0: POS, p1: POS, ID):
    center = (p0 + p1) / 2
    size = (p1 - p0)
    plot.text(p0.x, center.y, fr"{size.y:.3f}", verticalalignment='center', color="white", fontsize=8, rotation=90)
    plot.text(center.x, p0.y, fr"{size.x:.3f}", horizontalalignment='center', color="white", fontsize=8)
    plot.text(*center, f"{ID}", verticalalignment='center', horizontalalignment='center', color="white", fontsize=8)


def standard_draw_plan(plans:list[ProtoPlan], is_debug=False, task_id=None,text="",need_container=True):
    if task_id is None:
        task_id= str(uuid.uuid4())[0:8]

    fig, ax = plot.subplots(figsize=(16, 12))
    # img_data = []
    plot.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    plot.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    task_date = datetime.now().strftime("%Y%m%d")
    task_YM, task_dd = task_date[:6], task_date[6:]
    save_path = f"{PROCESSING_STEP_PATH if is_debug else SOLUTIONS_PATH}/{task_YM}/{task_dd}"
    os.makedirs(save_path, exist_ok=True)

    for i in range(len(plans)):
        # s = output.solution[i]
        # info = s.info
        # plans = s.plans
        plan=plans[i]

        msg = f"{text},id={plan.ID},item_count={len(plan.item_sequence)},利用率:{round(plan.util_rate() * 100, 2)}%,\n"
        frame = patches.Rectangle((0, 0), plan.material.width , plan.material.height,
                                  edgecolor='b', facecolor='gray')
        # ax.xaxis.tick_top()
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, plan.material.width + 0.1)
        ax.set_ylim(-0.1, plan.material.height + 0.1)
        ax.add_patch(frame)
        # ax.invert_yaxis()

        for item in plan.item_sequence:
            size = item.size
            pos = item.pos
            item_image_rect = patches.Rectangle(pos.to_tuple(),size.width,size.height, edgecolor='g', facecolor='darkgreen')
            ax.add_patch(item_image_rect)
            # _draw_text(plot, size.start, size.end, plan.ID + 1)
        ax.set_title(msg)

        # if is_debug:
        if need_container:
            for corner in plan.get_remain_containers():
                # corner_start, corner_end = POS(*corner[0]) * 0.001, POS(*corner[1]) * 0.001
                r= corner.rect
                # corner_size = (corner_end - corner_start)
                corner_rect = patches.Rectangle(r.start.to_tuple(), r.width,r.height, edgecolor='r',
                                                # facecolor='y',
                                                alpha=0.5
                                                )
                ax.add_patch(corner_rect)
            # _draw_text(plot, r.start, r.end, "")
        timestamp = round(time.time(),3).__str__()[5:]
        save_name = f"{save_path}/{task_id}_{plan.ID}{'_'+timestamp if is_debug else''}.png"
        plot.savefig(save_name)

        print(f"\rdraw task_id={task_id} plan_{plan.ID} ok", end="", flush=True)
        ax.cla()
    plot.close()
    #

if __name__ == '__main__':
    pass
