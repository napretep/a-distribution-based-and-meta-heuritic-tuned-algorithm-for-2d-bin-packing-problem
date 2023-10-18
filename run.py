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

from algorithm import Skyline,Distribution
from constant import *
from visualizing.draw_plan import standard_draw_plan

if __name__ == "__main__":
    np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(随机_data.shape[0], 300)
    data = 随机_data[data_idx]
    s = Skyline(data)
    d = Distribution(data)
    d.scoring_sys.parameters = [-3.13492378, -1.11457937 , 1.22778879,  4.83960852,  1.45576684,  7.38295868,
 -8.51840606,  3.56529697 , 4.10378123, -8.85677593, -3.35759839, -4.90079062,
  8.84279645,  4.77418402, -3.43442509, -8.48811202, -4.73466265,  7.90226931]
    d.scoring_sys.version = d.scoring_sys.V.GA
    s.run()
    d.run()
    print(f"skyline util rate = {s.avg_util_rate()},dist util rate={d.avg_util_rate()}")

    standard_draw_plan(d.solution,task_id=d.task_id)
    print("")
    standard_draw_plan(s.solution, task_id=s.task_id)
    pass