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
if __name__ == "__main__":
    current_path = os.getcwd()

    for filename in os.listdir(current_path):
        # 判断文件是否以 .npy 结尾
        if filename.endswith('.npy'):
            # 使用正则表达式匹配时间戳
            match = re.search(r'\(1', filename)
            if match:
                # 去掉时间戳部分
                new_filename = filename[:match.start()+1]+'0' + filename[match.start()+2:]
                # 重命名文件
                os.rename(os.path.join(current_path, filename), os.path.join(current_path, new_filename))
                print(f'File {filename} has been renamed to {new_filename}')
    pass