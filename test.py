# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = '可视化_查看数据分布.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 4:23'
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 假设我们有一组从正态分布中抽取的样本数据
data = np.random.normal(loc=0, scale=1, size=1000)



if __name__ == "__main__":
    import numpy as np
    from sklearn.neighbors import KernelDensity

    # 生成一些随机数据
    np.random.seed(0)
    X = np.random.normal(size=(100, 2))

    # 进行KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    import matplotlib.pyplot as plt

    # 生成新的样本
    new_X = kde.sample(1000)

    # 绘制散点图
    plt.scatter(new_X[:, 0], new_X[:, 1], s=5)
    plt.show()