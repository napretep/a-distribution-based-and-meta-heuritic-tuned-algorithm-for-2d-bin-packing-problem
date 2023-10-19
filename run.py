# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'run.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 4:23'
"""
from time import time

import numpy as np

from algorithm import Skyline,Distribution,MaxRect
from constant import *
from visualizing.draw_plan import standard_draw_plan

if __name__ == "__main__":
    for i in range(10):
        np.random.seed(int(time() * 10000) % 4294967296)
        data_idx = np.random.choice(华为杯_data.shape[0], 300)
        data = 华为杯_data[data_idx]
        m = MaxRect(data)
        s = Skyline(data)
        d = Distribution(data)
        d.scoring_sys.parameters = [-12.764467729922428, -7.2807524490032804, -17.405272153673526, 11.62060943355495, -17.767676767373285, 13.498788968865574, -3.058679224306764, -17.380930383866435, -17.380008727391687, -19.579085902347263, 15.561194939767207, 2.310615782862815, -5.273339286206582, 1.6631169187587558, -1.906345802422087, -3.3207320056750733, -7.4098035553284936, 12.394940621852495] # 1.07
        d.scoring_sys.version = d.scoring_sys.V.GA
        m.run()
        s.run()
        d.run(is_debug=False)
        print(f"maxrect={m.avg_util_rate()},skyline={s.avg_util_rate()},dist={d.avg_util_rate()}")
        # standard_draw_plan(d.solution, task_id=d.task_id)

    # standard_draw_plan(d.solution,task_id=d.task_id)
    # print("")
    # standard_draw_plan(s.solution, task_id=s.task_id)
    pass