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








def add_title_on_jointplot():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 创建一些样例数据
    tips = sns.load_dataset("tips")

    # 创建联合图
    g = sns.jointplot(data=tips, x="total_bill", y="tip")

    # 添加标题
    g.ax_joint.set_title('Joint Plot of Total Bill and Tip', pad=70)

    plt.show()


def add_colorbar():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 创建一些随机数据
    np.random.seed(0)
    data = np.random.multivariate_normal([0, 0], [(1, .5), (.5, 1)], size=200)
    df = pd.DataFrame(data, columns=['x', 'y'])

    def joint_heatmap(x, y, **kwargs):
        # 创建一个二维直方图
        bins_x = np.histogram_bin_edges(x, bins='auto')
        bins_y = np.histogram_bin_edges(y, bins='auto')
        counts, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y])

        # 创建一个热图
        ax = plt.gca()
        p = ax.imshow(counts.T, origin='lower', cmap='viridis',
                      extent=[bins_x.min(), bins_x.max(), bins_y.min(), bins_y.max()],
                      aspect='auto', **kwargs)

        # 添加颜色条
        plt.colorbar(p, ax=ax)

    g = sns.jointplot(data=df, x='x', y='y', kind='scatter')
    g.plot_joint(joint_heatmap)
    plt.show()
if __name__ == "__main__":
    add_title_on_jointplot()