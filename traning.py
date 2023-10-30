# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'traning.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/19 8:31'
"""
from time import time

import numpy as np

from algorithm import Skyline,Distribution,MaxRect
from algorithm.distribution_based import ScoringSys
from constant import *
from visualizing.draw_plan import standard_draw_plan



def standard():
    print("start")
    start_time = time()
    np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(华为杯_data.shape[0], 300)
    data = 华为杯_data[data_idx]
    d = Distribution(data)
    best_ind, best_score, log = d.fit_DE()
    print(d.task_id)
    d.scoring_sys.parameters = best_ind
    d.run()
    end_time = time()
    print("time cost=", end_time - start_time)
    print(d.avg_util_rate())
    standard_draw_plan(d.solution, task_id=d.task_id)



def mixed():
    pass


if __name__ == "__main__":

    pass