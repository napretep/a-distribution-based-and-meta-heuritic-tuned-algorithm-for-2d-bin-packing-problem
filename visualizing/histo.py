# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = '二维图形判断.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 4:32'
"""
import seaborn as sns
import matplotlib.pyplot as plt

# 假设我们有两个变量 x 和 y
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [2, 3, 4, 3, 2, 4, 5, 6, 5]












if __name__ == "__main__":

    pass



    # 创建一个新的图形
    plt.figure(figsize=(12, 6))

    # 二维直方图
    plt.subplot(1, 3, 2)
    sns.histplot(x=x, y=y, bins=30, cbar=True)
    plt.title('2D Histogram')

    # 二维核密度估计图
    plt.subplot(1, 3, 3)
    sns.kdeplot(x=x, y=y, fill=True)
    plt.title('2D KDE plot')

    # 显示图形
    plt.tight_layout()
    plt.show()