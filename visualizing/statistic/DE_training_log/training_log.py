# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'training_log.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/6 18:14'
"""
import numpy as np
import matplotlib.pyplot as plt




def draw_log(log_path):

    data = np.load(log_path)

    # 创建一个新的图形
    plt.figure()

    # 绘制折线图
    plt.plot(data[:,0],data[:,1])

    # 显示图形
    plt.show()



if __name__ == "__main__":
    # draw_log("production_data1_300_traning_log__1699236802.npy")
    draw_log("production_data2_300_traning_log__1699253136.npy")
    pass