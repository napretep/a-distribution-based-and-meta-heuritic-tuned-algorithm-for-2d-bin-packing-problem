# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'show_compare.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/7 3:24'
"""
import numpy as np

from constant import *
def show_random():
    data = np.load("random_ratio(1,30)_MaxRect_production_data1_500_1699298678.7625933.npy")
    print(data.shape)
    new_li=[]
    for i in range(data.shape[0]):
        new_li.append(np.mean(data[i,:]))
    plt.plot(range(len(new_li)),new_li)
    plt.show()

def show_determ():
    data1 = np.load("standard_Dist_production_data1_500_1699300026.7130404.npy")
    data2 = np.load("standard_Skyline_production_data1_500_1699299050.4751542.npy")
    data3 = np.load("standard_MaxRect_production_data1_500_1699298991.4508817.npy")
    # print(data.shape)
    plt.boxplot([data1,data2,data3])
    plt.show()

if __name__ == "__main__":
    print(np.load("random_ratio(0,30)_Dist2_production_data1_3000_.npy").shape)
    pass