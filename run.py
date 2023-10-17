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


if __name__ == "__main__":
    np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(华为杯_data.shape[0], 300)
    data = 华为杯_data[data_idx]
    s = Skyline(data)
    d = Distribution(data)
    d.scoring_sys.parameters = [7.84605513, -0.62829151, -0.34513113, -6.99626145, 6.84425151, -0.4938427,
                                -3.40503507, 6.32845631, -4.60328576, 3.80204546, 3.65414871, -5.52031716,
                                5.63098865, 2.63530608, -2.66109476, -2.82704527, 5.19361657, 7.49876944]
    d.scoring_sys.version = d.scoring_sys.V.GA
    s.run()
    d.run()
    print(f"skyline util rate = {s.avg_util_rate()},dist util rate={d.avg_util_rate()}")


    pass