# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'clearn.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/9 21:38'
"""

import os
import re
from constant import *
# 当前路径


def remove_timestamp():
    current_path = os.getcwd()

    for filename in os.listdir(current_path):
        # 判断文件是否以 .npy 结尾
        if filename.endswith('.npy'):
            # 使用正则表达式匹配时间戳
            match = re.search(r'1699', filename)
            if match:
                # 去掉时间戳部分
                new_filename = filename[:match.start()] + '.npy'
                # 重命名文件
                os.rename(os.path.join(current_path, filename), os.path.join(current_path, new_filename))
                print(f'File {filename} has been renamed to {new_filename}')

def change_name():
    current_path = os.getcwd()

    for filename in os.listdir(current_path):
        if filename.endswith('.npy'):
            # match = re.search(r'random_ratio\(0,30\)', filename)
            # if match:
                # data_type = re.findall(r'random_ratio\(0,30\)',filename)[0]
            algo_name = re.findall(r"MaxRect|Skyline",filename)[0]
            data_set_name = re.findall(f"{PRODUCTION_DATA1}|{PRODUCTION_DATA2}|{RANDOMGEN_DATA}",filename)[0]
            scale = re.findall(r"100_\.npy|300_\.npy|500_\.npy|1000_\.npy|3000_\.npy|5000_\.npy",filename)[0]

            new_filename = f"{STANDARD}_{data_set_name}_{algo_name}_{scale}_.npy"
            print(new_filename)
            os.rename(os.path.join(current_path, filename),os.path.join(current_path,new_filename))

def change_name2():
    current_path = os.getcwd()

    for filename in os.listdir(current_path):
        if filename.endswith('.npy'):
            match = re.search(r'random_data', filename)
            if match:
                new_filename = re.sub(r"random_data","randomGen_data",filename)
                print(new_filename)
                os.rename(os.path.join(current_path, filename), os.path.join(current_path, new_filename))

def change_name3():
    current_path = os.getcwd()

    for filename in os.listdir(current_path):
        if filename.endswith('.npy'):

            new_filename = re.sub(r"_\.npy_\.npy","_.npy",filename)
            print(new_filename)
            os.rename(os.path.join(current_path, filename), os.path.join(current_path, new_filename))

if __name__ == "__main__":
    change_name2()
    pass