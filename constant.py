# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'constant.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 4:23'
"""
import dataclasses
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *
import inspect
PROJECT_ROOT_PATH = os.path.split(__file__)[0]

DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')

IMAGES_PATH = os.path.join(DATA_PATH, 'images')

PROCESSING_STEP_PATH = os.path.join(IMAGES_PATH, 'processing_step')

SOLUTIONS_PATH = os.path.join(IMAGES_PATH, 'solutions')

# 华为_data = os.path.join(DATA_PATH, '华为杯数据')

__all__ = ['外包_data', '华为杯_data', "COL", "随机_data", "PROCESSING_STEP_PATH", "SOLUTIONS_PATH","Rect", "ProtoPlan", "POS","Item","Line"]


class COL:
    maxL = 1
    minL = maxL + 1
    length = minL + 1
    width = length + 1
    Remain = 6
    ID = 0

    class Material:
        pass

    class Item:
        Texture = 7

    class Edge:
        Left = 0
        Bottom = 1
        Right = 2
        Top = 3


@dataclasses.dataclass
class POS:
    x: int | float = 0
    y: int | float = 0

    def __sub__(self, other: "POS|float|int"):
        if isinstance(other, POS):
            return POS(self.x - other.x, self.y - other.y)
        elif isinstance(other, (float, int)):
            return POS(self.x - other, self.y - other)
        else:
            return NotImplemented

    def __truediv__(self, scalar: "int|float"):
        if isinstance(scalar, (int, float)):
            return POS(self.x / scalar, self.y / scalar)
        else:
            return NotImplemented

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return POS(self.x * scalar, self.y * scalar)
        else:
            return NotImplemented

    def __add__(self, other: "POS|float|int"):
        if isinstance(other, POS):
            return POS(self.x + other.x, self.y + other.y)
        elif isinstance(other, (float, int)):
            return POS(self.x + other, self.y + other)
        else:
            return NotImplemented

    def __iter__(self):
        return iter((self.x, self.y))

    def to_tuple(self):
        return self.x, self.y

    def __eq__(self, other:"POS"):
        return self.x == other.x and self.y == other.y

    def __gt__(self, other: "POS"):
        return self.x > other.x and self.y > other.y

    def __lt__(self, other: "POS"):
        return self.x < other.x and self.y < other.y

    def __ge__(self, other:"POS"):
        return self.x >= other.x and self.y >= other.y
    def __le__(self, other:"POS"):
        return self.x <= other.x and self.y <= other.y


@dataclasses.dataclass
class Line:
    start:POS
    end:POS

@dataclasses.dataclass
class Rect:
    start:POS
    end:POS
    ID:int|None=None

    def __init__(self, *args):
        if len(args) == 2:
            if type(args[0])==POS:
                self.start = args[0]
                self.end = args[1]
            else:
                self.start,self.end=POS(*args[0]),POS(*args[1])
        elif len(args)==4:
            self.start = POS(*args[:2])
            self.end = POS(*args[2:])
        elif len(args)==3:
            if type(args[0])==POS:
                self.start = args[0]
                self.end = POS(*args[1:])
            elif type(args[2])==POS:
                self.start = POS(*args[:2])
                self.end = args[2]
        elif len(args)==0:
            self.start = self.end = POS(0,0)
        if self.start>=self.end:
            new_end = self.start
            self.start = self.end
            self.end= new_end

    @property
    def center(self):
        return (self.start + self.end) / 2
    @property
    def topLeft(self):
        return POS(self.start.x,self.end.y)
    @property
    def topRight(self):
        return self.end
    @property
    def bottomLeft(self):
        return self.start
    @property
    def bottomRight(self):
        return POS(self.end.x,self.start.y)
    @property
    def width(self)->int|float:
        return self.end.x - self.start.x
    @property
    def height(self)->int|float:
        return self.end.y - self.start.y
    @property
    def area(self):
        return self.width * self.height

    def transpose(self):
        """转置"""
        # end = POS(self.start+self.width, self.start+self.height)
        end = POS(self.height,self.width)+self.start
        return Rect(self.start, end,self.ID)

    def copy(self):
        return Rect(self.start,self.end,self.ID)

    def __eq__(self, other:"Type[Rect|Line|POS]|Rect"):
        if type(other)==Rect:
            return self.start == other.start and self.end == other.end
        elif inspect.isclass(other):
            if other is Rect:
                l = self.end-self.start
                return l.x>0 and l.y>0
            elif other is Line:
                l = self.end - self.start
                return (l.x>0 and l.y==0) or (l.x==0 and l.y>0)
            elif other is POS:
                return self.end == self.start
            else:
                raise ValueError("other must be Rect or Line or POS")

    def __sub__(self, other:POS|float|int):
        return Rect(self.start - other, self.end - other)

    def __add__(self, other:POS|float|int):
        return Rect(self.start + other, self.end + other)

    def __contains__(self, item:"Rect|POS"):
        if type(item)==Rect:
            return self.start <= item.start and self.end >= item.end
        elif type(item)==POS:
            return self.start <= item.start and self.end >= item.end
        else:
            raise ValueError("item must be Rect or POS")
    def __and__(self, other:"Rect"):
        """交"""
        if other in self:
            return Rect(other.start, other.end)
        elif self in other:
            return Rect(self.start, self.end)

        else:

            if other.bottomLeft in self:
                # 下左,下右
                if other.bottomRight in self:
                    return Rect(other.bottomLeft,other.bottomRight.x,self.topLeft.y)
                # 下左,上左
                elif other.topLeft in self:
                    return Rect(other.bottomLeft,self.topLeft.x,other.topRight.y)
                else:
                    return Rect(other.bottomLeft,self.topLeft)
            elif other.topRight in self:
                # 上右,上左
                if other.topLeft in self:
                    return Rect(other.bottomLeft.x,self.bottomLeft.y,other.topLeft)
                # 上右,下右
                elif other.bottomRight in self:
                    return Rect(self.bottomRight.x,other.bottomRight.y,other.topRight)
                else:
                    return Rect(self.bottomLeft, other.topRight)
            elif other.topLeft in self:
                # 上左,上右
                if other.topRight in self:
                    return Rect(other.bottomLeft.x,self.bottomLeft.y,other.topLeft)
                # 上左,下左
                elif other.bottomLeft in self:
                    return Rect(other.bottomLeft,self.topLeft.x,other.topRight.y)
                else:
                    return Rect(other.bottomLeft.x,self.bottomLeft.y,self.topRight.x,other.topRight.y)
            elif other.bottomRight in self:
                # 下右,下左
                if other.bottomLeft in self:
                    return Rect(other.bottomLeft,other.bottomRight.x,self.topLeft.y)
                # 下右,上右
                elif other.topRight in self:
                    return Rect(self.bottomRight.x,other.bottomRight.y,other.topRight)
                else:
                    return Rect(self.bottomRight.x,other.bottomRight.y,self.bottomRight.x,self.topRight.y)
            elif other.topRight.x<=self.topRight.x and other.topLeft.x>=self.topLeft.x:
                return Rect(other.bottomLeft.x,self.bottomLeft.y,other.topLeft.x,self.topLeft.y)
            elif other.topLeft.y<=self.topLeft.y and other.bottomRight.y>=self.bottomRight.y:
                return Rect(self.bottomLeft.x,other.bottomLeft.y,self.topRight.x,other.topRight.y)
            else:
                return Rect()


    def __bool__(self):
        if self.start == self.end:
            return False
        else:
            return True

    # def contain(self, item:"Rect"):
    #     return (item+self.start) in self




class Item:
    ID:int
    size:Rect
    pos:POS
    def transpose(self):
        return Item(self.ID,self.size.transpose())

    def __eq__(self, other:"Item"):
        return self.ID==other.ID

@dataclasses.dataclass
class ProtoPlan:
    """

    """
    ID:int
    material:Rect
    item_sequence:list[Item]
    remain_containers:"list[Container]|None"=None

    def util_rate(self):
        return sum([item.size.area for item in self.item_sequence]) / self.material.area
# @dataclasses.dataclass
# class Solution:
#     plan_list:list[ProtoPlan]

_temp_外包_data = np.loadtxt(os.path.join(DATA_PATH, r'外包数据\items.csv'), delimiter=',')
外包_data: "np.ndarray|None" = None
# _temp_外包_data = np.column_stack((_temp_外包_data[:,0],np.maximum(_temp_外包_data[:,1],_temp_外包_data[:,2]),np.minimum(_temp_外包_data[:,1],_temp_外包_data[:,2])))
for i in range(_temp_外包_data.shape[0]):
    row = _temp_外包_data[i]
    item_count = row[COL.Remain]
    new_rows = np.repeat(row[np.newaxis, :], item_count, axis=0)

    if 外包_data is None:
        外包_data = new_rows
    else:
        外包_data = np.row_stack((外包_data, new_rows))

maxL_max = np.max(外包_data[:, COL.maxL])
maxL_min = np.min(外包_data[:, COL.maxL])
minL_max = np.max(外包_data[:, COL.minL])
minL_min = np.min(外包_data[:, COL.minL])
外包_data = np.column_stack(
        (外包_data[:, 0],
         (外包_data[:, COL.maxL] - maxL_min) / (maxL_max - maxL_min),
         (外包_data[:, COL.minL] - minL_min) / (minL_max - minL_min)))
外包_data = np.column_stack(
        (外包_data[:, 0], np.maximum(外包_data[:, 1], 外包_data[:, 2]), np.minimum(外包_data[:, 1], 外包_data[:, 2]))
)

华为杯_data: "np.ndarray|None" = None
# item_id, item_material, item_num, item_length, item_width, item_order
for i in range(5):
    _temp_华为杯_data = np.loadtxt(os.path.join(DATA_PATH, r'华为杯数据\data' + str(i + 1) + '.csv'), delimiter=',')
    if 华为杯_data is None:
        华为杯_data = _temp_华为杯_data
    else:
        华为杯_data = np.row_stack((华为杯_data, _temp_华为杯_data))
华为杯_data = np.column_stack((华为杯_data[:, 0], np.maximum(华为杯_data[:, 1], 华为杯_data[:, 2]), np.minimum(华为杯_data[:, 1], 华为杯_data[:, 2])))
# 华为杯_data = np.column_stack((华为杯_data[:,0],华为杯_data[:,1],华为杯_data[:,2]))

maxL_max = np.max(华为杯_data[:, COL.maxL])
maxL_min = np.min(华为杯_data[:, COL.maxL])
minL_max = np.max(华为杯_data[:, COL.minL])
minL_min = np.min(华为杯_data[:, COL.minL])
华为杯_data = np.column_stack(
        (华为杯_data[:, 0],
         (华为杯_data[:, COL.maxL] - maxL_min) / (maxL_max - maxL_min),
         (华为杯_data[:, COL.minL] - minL_min) / (minL_max - minL_min)))

# 设置随机种子以保证结果可复现
np.random.seed(0)

mu, sigma = 0.1, 50  # 设置均值和标准差
x_samples = np.random.normal(mu, sigma, 1000)
y_samples = np.random.normal(mu, sigma, 1000)
samples1 = np.column_stack((x_samples, y_samples))
mu, sigma = 0.1, 0.1  # 设置均值和标准差
x_samples = np.random.normal(mu, sigma, 500)
y_samples = np.random.normal(mu, sigma, 500)
samples2 = np.column_stack((x_samples, y_samples))
mu, sigma = 0.1, 0.1  # 设置均值和标准差
x_samples = np.random.normal(mu, sigma, 500)
y_samples = np.random.normal(mu + 0.2, sigma, 500)
samples3 = np.column_stack((x_samples, y_samples))
x_samples = np.random.uniform(0, 1, 200)
y_samples = np.random.uniform(0, 1, 200)
samples4 = np.column_stack((x_samples, y_samples))
samples = np.row_stack((samples2, samples1, samples3, samples4))
samples = samples[(0 <= samples[:, 0]) & (samples[:, 0] <= 1) & (0 <= samples[:, 1]) & (samples[:, 1] <= 1)]

随机_data = np.column_stack((np.zeros(samples.shape[0]), np.maximum(samples[:, 0], samples[:, 1]), np.minimum(samples[:, 0], samples[:, 1])))




def unify(data:"np.ndarray"):
    """单位化"""


# 随机_data = np.column_stack((np.zeros(samples.shape[0]),samples))

def kde(data):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    # ax.set_aspect('equal')
    sns.kdeplot(x=data[:, 1], y=data[:, 2], fill=True, bw_adjust=1)
    plt.show()
    pass


if __name__ == "__main__":
    print(Rect(POS(10,10),POS(10,0))==Rect)

    pass
