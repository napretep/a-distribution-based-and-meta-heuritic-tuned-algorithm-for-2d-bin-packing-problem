# -*- coding: utf-8 -*-
"""
__project_ = '代码'
__file_name__ = 'constant.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/10/2 4:23'
"""
import dataclasses
import os, json
import uuid
from time import time

import BinPacking2DAlgo
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *
import inspect

from scipy.stats import gaussian_kde
import pandas as pd

PROJECT_ROOT_PATH = os.path.split(os.path.abspath(__file__))[0]

PARAM_PATH = os.path.join(PROJECT_ROOT_PATH,"param")
SYNC_PATH = json.load(open(os.path.join(PROJECT_ROOT_PATH, "config.json"), "r", encoding="utf-8"))["data_sync_path"]

DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')

IMAGES_PATH = os.path.join(DATA_PATH, 'images')

PROCESSING_STEP_PATH = os.path.join(IMAGES_PATH, 'processing_step')

SOLUTIONS_PATH = os.path.join(IMAGES_PATH, 'solutions')

RANDOMGEN_DATA = "randomGen_data"
PRODUCTION_DATA1 = "production_data1"
PRODUCTION_DATA2 = "production_data2"
OPEN_ACCESS_DATA1="open_access_data1"
OPEN_ACCESS_DATA2="open_access_data2"


NOISED = "noised"
STANDARD = "standard"
DATA_NAME="data_name"
DATA_CONTAINER="data_container"


DATA_SCALES = (100, 300, 500, 1000, 3000, 5000)
RUN_COUNT = 40
MATERIAL_SIZE = (2440, 1220)


# 华为_data = os.path.join(DATA_PATH, '华为杯数据')

# __all__ = ['外包_data', '华为杯_data', "COL", "随机_data", "PROCESSING_STEP_PATH", "SOLUTIONS_PATH", "Rect", "ProtoPlan", "POS", "Item", "Line", "Container", "unify", "Algo", "DATA_SCALES", "RUN_COUNT", "MATERIAL_SIZE", "random_choice",
#            "kde_sample"]


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

    def copy(self):
        return POS(self.x, self.y)

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

    def __eq__(self, other: "POS"):
        return self.x == other.x and self.y == other.y

    def __gt__(self, other: "POS"):
        return self.x > other.x and self.y > other.y

    def __lt__(self, other: "POS"):
        return self.x < other.x and self.y < other.y

    def __ge__(self, other: "POS"):
        return self.x >= other.x and self.y >= other.y

    def __le__(self, other: "POS"):
        assert type(other) == POS
        return self.x <= other.x and self.y <= other.y


@dataclasses.dataclass
class Line:
    start: POS
    end: POS


@dataclasses.dataclass
class Rect:
    start: POS
    end: POS
    ID: int | None = None

    def __init__(self, *args, ID=None):
        self.ID = ID
        if len(args) == 2:
            if type(args[0]) == POS:
                self.start = args[0]
                self.end = args[1]
            else:
                self.start, self.end = POS(*args[0]), POS(*args[1])
        elif len(args) == 4:
            self.start = POS(*args[:2])
            self.end = POS(*args[2:])
        elif len(args) == 3:
            if type(args[0]) == POS:
                self.start = args[0]
                self.end = POS(*args[1:])
            elif type(args[2]) == POS:
                self.start = POS(*args[:2])
                self.end = args[2]
        elif len(args) == 0:
            self.start = self.end = POS(0, 0)
        if self.start >= self.end:
            new_end = self.start
            self.start = self.end
            self.end = new_end

    def __sub__(self, other: "Rect"):
        assert type(other) == Rect
        inter_rect = self & other

        # 判断新加入的矩形是否和其他空余矩形有交集,如果有,则剔除这个矩形,同时生成它的剩余部分作为新的空余矩形
        if inter_rect == Rect:
            result = [None, None, None, None]  # 上下左右
            top_c, bottom_c, left_c, right_c, = None, None, None, None
            # 上部
            if inter_rect.topRight.y < self.topRight.y:
                top_c = Rect(POS(self.topLeft.x, inter_rect.topLeft.y), self.topRight, )
                if top_c == Rect:  # 还需要判断新容器是个有面积的矩形
                    result[0] = top_c
            # 下部
            if inter_rect.bottomRight.y > self.bottomRight.y:
                bottom_c = Rect(self.bottomLeft, POS(self.bottomRight.x, inter_rect.bottomRight.y), )
                if bottom_c == Rect:
                    result[1] = bottom_c
            # 右部
            if inter_rect.topRight.x < self.topRight.x:
                left_c = Rect(POS(inter_rect.bottomRight.x, self.bottomLeft.y), self.topRight, )
                if left_c == Rect:
                    result[2] = left_c
            # 左部
            if inter_rect.topLeft.x > self.topLeft.x:
                right_c = Rect(self.bottomLeft, POS(inter_rect.topLeft.x, self.topRight.y), )
                if right_c == Rect:
                    result[3] = right_c
            return result
        else:
            return self, None, None, None

    @property
    def center(self):
        return (self.start + self.end) / 2

    @property
    def topLeft(self):
        return POS(self.start.x, self.end.y)

    @property
    def topRight(self):
        return self.end

    @property
    def bottomLeft(self):
        return self.start

    @property
    def bottomRight(self):
        return POS(self.end.x, self.start.y)

    @property
    def size(self):
        return POS(self.width, self.height)

    @property
    def width(self) -> int | float:
        return self.end.x - self.start.x

    @property
    def height(self) -> int | float:
        return self.end.y - self.start.y

    @property
    def area(self):
        return self.width * self.height

    def transpose(self):
        """转置"""
        # end = POS(self.start+self.width, self.start+self.height)
        end = POS(self.height, self.width) + self.start
        r = Rect(self.start, end)
        r.ID = self.ID
        return r

    def copy(self):
        return Rect(self.start, self.end, ID=self.ID)

    def __eq__(self, other: "Type[Rect|Line|POS]|Rect"):
        if type(other) == Rect:
            return self.start == other.start and self.end == other.end
        elif inspect.isclass(other):
            if other is Rect:
                l = self.end - self.start
                return l.x > 0 and l.y > 0
            elif other is Line:
                l = self.end - self.start
                return (l.x > 0 and l.y == 0) or (l.x == 0 and l.y > 0)
            elif other is POS:
                return self.end == self.start
            else:
                raise ValueError("other must be Rect or Line or POS")

    def __sub__(self, other: POS | float | int):
        return Rect(self.start - other, self.end - other)

    def __add__(self, other: POS | float | int):
        return Rect(self.start + other, self.end + other)

    def __mul__(self, other):
        assert type(other) in (int, float)
        return Rect(self.start * other, self.end * other, ID=self.ID)

    def __contains__(self, item: "Rect|POS"):
        if type(item) == Rect:
            return self.start <= item.start and self.end >= item.end
        elif type(item) == POS:
            return self.start <= item and self.end >= item
        else:
            raise ValueError("item must be Rect or POS")

    def __and__(self, other: "Rect"):
        """交"""
        if other in self:
            return Rect(other.start, other.end)
        elif self in other:
            return Rect(self.start, self.end)
        else:

            if other.bottomLeft in self:
                # 下左,下右
                if other.bottomRight in self:
                    return Rect(other.bottomLeft, other.bottomRight.x, self.topLeft.y)
                # 下左,上左
                elif other.topLeft in self:
                    return Rect(other.bottomLeft, self.topRight.x, other.topRight.y)
                else:
                    return Rect(other.bottomLeft, self.topRight)
            elif other.topRight in self:
                # 上右,上左
                if other.topLeft in self:
                    return Rect(other.bottomLeft.x, self.bottomLeft.y, other.topRight)
                # 上右,下右
                elif other.bottomRight in self:
                    return Rect(self.bottomLeft.x, other.bottomRight.y, other.topRight)
                else:
                    return Rect(self.bottomLeft, other.topRight)
            elif other.topLeft in self:
                # 上左,上右
                if other.topRight in self:
                    return Rect(other.bottomLeft.x, self.bottomLeft.y, other.topRight)
                # 上左,下左
                elif other.bottomLeft in self:
                    return Rect(other.bottomLeft, self.topRight.x, other.topRight.y)
                else:
                    return Rect(other.bottomLeft.x, self.bottomLeft.y, self.topRight.x, other.topRight.y)
            elif other.bottomRight in self:
                # 下右,下左
                if other.bottomLeft in self:
                    return Rect(other.bottomLeft, other.bottomRight.x, self.topLeft.y)
                # 下右,上右
                elif other.topRight in self:
                    return Rect(self.bottomLeft.x, other.bottomRight.y, other.topRight)
                else:
                    # 仅一个右下点在里面
                    return Rect(self.topLeft.x, other.bottomLeft.y, other.bottomRight.x, self.topRight.y)
            elif self.topLeft.x <= other.topLeft.x and other.topRight.x <= self.topRight.x and \
                    other.bottomRight.y <= self.bottomRight.y and self.topLeft.y <= other.topLeft.y:
                return Rect(other.bottomLeft.x, self.bottomLeft.y, other.topRight.x, self.topRight.y)
            elif self.bottomRight.y <= other.bottomRight.y and other.topLeft.y <= self.topLeft.y \
                    and other.topLeft.x <= self.topLeft.x and self.topRight.x <= other.topRight.x:
                return Rect(self.bottomLeft.x, other.bottomLeft.y, self.topRight.x, other.topRight.y)
            else:
                return Rect()

    def __str__(self):
        return f"Rect(({self.start.x},{self.start.y}),({self.end.x},{self.end.y}))"

    def __bool__(self):
        if self.start == self.end:
            return False
        else:
            return True

    def __iter__(self):
        return iter((self.start, self.end))
    # def contain(self, item:"Rect"):
    #     return (item+self.start) in self


@dataclasses.dataclass
class Container:
    rect: Rect
    plan_id: int = None

    def __init__(self, start: POS, end: POS, plan_id: "int|None" = None):
        self.rect = Rect(start, end)
        self.plan_id = plan_id

    def __eq__(self, other: "Container|Rect|Line|POS"):
        # assert type(other) == Container
        if type(other) == Container:
            return self.rect == other.rect
        else:
            return self.rect == other

    def __contains__(self, item):
        assert type(item) == Rect
        return item in self.rect

    def __repr__(self):
        return self.rect.__str__()

    def __str__(self):
        return self.__repr__()


@dataclasses.dataclass
class Item:
    ID: int
    size: Rect
    pos: POS

    def transpose(self):
        return Item(self.ID, self.size.transpose(), self.pos)

    @property
    def rect(self):
        return self.size + self.pos

    def copy(self):
        return Item(self.ID, self.size, self.pos)

    def __eq__(self, other: "Item"):
        return self.ID == other.ID


@dataclasses.dataclass
class ProtoPlan:
    """

    """
    ID: int
    material: Rect
    item_sequence: list[Item]
    remain_containers: "list[Container]|None" = None

    def util_rate(self):
        return sum([item.size.area for item in self.item_sequence]) / self.material.area

    def get_remain_containers(self):
        if self.remain_containers is not None:
            return self.remain_containers
        else:
            raise NotImplementedError()


def make_data_from_json(data_name,filecount):
    x_data = []
    z_data_path = lambda i: os.path.join(DATA_PATH, data_name, f"{i}.json")
    for i in range(1, filecount+1):
        temp_aa_data = json.load(open(z_data_path(i)))
        temp_aa_old_material = temp_aa_data["Objects"][0]
        material_width, material_height = temp_aa_old_material["Length"], temp_aa_old_material["Height"]
        for item in temp_aa_data["Items"]:
            new_width, new_height = item["Length"] / material_width * MATERIAL_SIZE[0], item["Height"] / material_height * MATERIAL_SIZE[1]
            item_li = [[int(max(new_width, new_height)), int(min(new_height, new_width))] for i in range(item["Demand"])]
            x_data += item_li
    x_data = np.column_stack((range(len(x_data)), np.array(x_data)))
    np.save(os.path.join(DATA_PATH, data_name, "data.npy"),x_data)


def make_wb_data():
    外包_data: "np.ndarray|None" = None
    _temp_外包_data = np.loadtxt(os.path.join(DATA_PATH, r'wb_data\items.csv'), delimiter=',')
    for i in range(_temp_外包_data.shape[0]):
        row = _temp_外包_data[i]
        item_count = row[COL.Remain]
        new_rows = np.repeat(row[np.newaxis, :], item_count, axis=0)

        if 外包_data is None:
            外包_data = new_rows
        else:
            外包_data = np.row_stack((外包_data, new_rows))

    外包_data = np.column_stack(
            (外包_data[:, 0], np.maximum(外包_data[:, 1], 外包_data[:, 2]), np.minimum(外包_data[:, 1], 外包_data[:, 2]))
    )
    外包_data_idx = (外包_data[:, 1] <= MATERIAL_SIZE[0]) & (外包_data[:, 2] <= MATERIAL_SIZE[1])
    外包_data = 外包_data[外包_data_idx]
    np.save(os.path.join(DATA_PATH, r'wb_data',"data.npy"),外包_data)

def make_hw_data():
    华为杯_data: "np.ndarray|None" = None
    # item_id, item_material, item_num, item_length, item_width, item_order
    for i in range(5):
        _temp_华为杯_data = np.loadtxt(os.path.join(DATA_PATH, r'hw_data\data' + str(i + 1) + '.csv'), delimiter=',')
        if 华为杯_data is None:
            华为杯_data = _temp_华为杯_data
        else:
            华为杯_data = np.row_stack((华为杯_data, _temp_华为杯_data))
    华为杯_data = np.column_stack((华为杯_data[:, 0], np.maximum(华为杯_data[:, 1], 华为杯_data[:, 2]), np.minimum(华为杯_data[:, 1], 华为杯_data[:, 2])))
    np.save(os.path.join(DATA_PATH, r'hw_data',"data.npy"),华为杯_data)


aa_data = np.load(os.path.join(DATA_PATH, "AA_data", "data.npy"))

cz_data = np.load(os.path.join(DATA_PATH, "CZ_data", "data.npy"))

华为杯_data = np.load(os.path.join(DATA_PATH, "hw_data", "data.npy"))

外包_data = np.load(os.path.join(DATA_PATH, "wb_data", "data.npy"))





# _temp_外包_data = np.column_stack((_temp_外包_data[:,0],np.maximum(_temp_外包_data[:,1],_temp_外包_data[:,2]),np.minimum(_temp_外包_data[:,1],_temp_外包_data[:,2])))







# 设置随机种子以保证结果可复现
np.random.seed(0)

mu, sigma = 0.1, 2  # 设置均值和标准差
x_samples = np.random.normal(mu + 1, sigma, 1000)
y_samples = np.random.normal(mu, sigma, 1000)
samples1 = np.column_stack((x_samples, y_samples))
mu, sigma = 0.1, 0.1  # 设置均值和标准差
x_samples = np.random.normal(mu, sigma, 1000)
y_samples = np.random.normal(mu, sigma, 1000)
samples2 = np.column_stack((x_samples, y_samples))
mu, sigma = 0.1, 0.1  # 设置均值和标准差
x_samples = np.random.normal(mu, sigma, 500)
y_samples = np.random.normal(mu + 0.2, sigma, 500)
samples3 = np.column_stack((x_samples, y_samples))
x_samples = np.random.uniform(0, 1, 200)
y_samples = np.random.uniform(0, 1, 200)
samples4 = np.column_stack((x_samples, y_samples))
x_samples = np.random.uniform(0, 1, 200)
y_samples = np.random.uniform(0.45, 0.65, 200)
samples5 = np.column_stack((x_samples, y_samples))
x_samples = np.random.uniform(0, 1, 200)
y_samples = np.random.uniform(0.25, 0.4, 200)
samples6 = np.column_stack((x_samples, y_samples))
mu, sigma = 0.1, 0.1  # 设置均值和标准差
x_samples = np.random.normal(0.327, 1, 500)
y_samples = np.random.normal(0.2, 1, 500)
samples7 = np.column_stack((x_samples, y_samples))
samples = np.row_stack((samples2, samples1, samples3, samples4, samples5, samples6, samples7))
samples = samples[(0 <= samples[:, 0]) & (samples[:, 0] <= 1) & (0 <= samples[:, 1]) & (samples[:, 1] <= 1)]

随机_data: "np.ndarray" = np.column_stack((np.zeros(samples.shape[0]), np.maximum(samples[:, 0], samples[:, 1]) * MATERIAL_SIZE[0], np.minimum(samples[:, 0], samples[:, 1]) * MATERIAL_SIZE[1]))

随机_data = 随机_data.astype(int)


def unify(data: "np.ndarray"):
    """单位化
    :param:data:[ID,col1,col2]
    """

    col1_idx = 1
    col2_idx = 2
    data1 = np.column_stack((data[:, 0], np.maximum(data[:, col1_idx], data[:, col2_idx]), np.minimum(data[:, col1_idx], data[:, col2_idx])))
    max_data = np.max(data1[:, col1_idx:col2_idx + 1])
    min_data = np.min(data1[:, col1_idx:col2_idx + 1])
    data_col1 = data1[:, 0]
    data_col2 = data1[:, 1]
    final_data = np.column_stack((data1[:, 0], ((data_col1 - min_data) / (max_data - min_data)), ((data_col2 - min_data) / (max_data - min_data))))

    return final_data


def random_choice(data, count=5000, true_random=True):
    if true_random:
        np.random.seed(int(time() * 10000) % 4294967296)
    data_idx = np.random.choice(data.shape[0], count, replace=False)
    return data[data_idx]


def random_mix(data: "np.ndarray", random_ratio):
    """
    data:shape=(n,2)
    random_ratio : (0,0.3)
    """
    random_item_scale = int(np.random.uniform(*random_ratio) * data.shape[0])
    determ_data_idx = np.random.choice(data.shape[0], size=data.shape[0] - random_item_scale, replace=False)
    determ_data = data[determ_data_idx]
    random_x = (np.random.uniform(0.1, 1, random_item_scale) * MATERIAL_SIZE[0]).astype(int)
    random_y = (np.random.uniform(0.1, 1, random_item_scale) * MATERIAL_SIZE[1]).astype(int)
    random_data = np.column_stack((random_x, random_y))
    result = np.row_stack((determ_data, random_data))
    result = np.column_stack((range(data.shape[0]), np.maximum(result[:, 0], result[:, 1]), np.minimum(result[:, 0], result[:, 1])))

    return result



class Algo:
    def __init__(self, item_data: "np.ndarray|list" = None, material_data: "Iterable" = MATERIAL_SIZE, task_id=None):
        self.items: "list[Item]" = [Item(ID=item[0],
                                         size=Rect(POS(0, 0), POS(item[1], item[2])),
                                         pos=POS(0, 0)
                                         ) for item in item_data] if item_data is not None else []
        self.material: "Rect" = Rect(POS(0, 0), POS(*material_data))
        self.solution: "list[ProtoPlan]|None" = []
        # self.min_size = min(np.min(item_data[:, COL.minL]), np.min(item_data[:, COL.maxL]))
        self.task_id = task_id if task_id else str(uuid.uuid4())[0:8]

    def load_data(self, item_data: "np.ndarray"):
        self.items: "list[Item]" = [Item(ID=item[0],
                                         size=Rect(POS(0, 0), POS(item[1], item[2])),
                                         pos=POS(0, 0)
                                         ) for item in item_data]
        # self.min_size = min(np.min(item_data[:, COL.minL]), np.min(item_data[:, COL.maxL]))

    def avg_util_rate(self):
        return sum([item.util_rate() for item in self.solution]) / len(self.solution)

    def run(self, is_debug=False):
        raise NotImplementedError()


def kde_sample(data, count=1000):  # epanechnikov,gaussian
    np.random.seed(int(time() * 10000) % 4294967296)
    kde_ = gaussian_kde((data[:, 1:3]).T, bw_method=0.1)
    resample_data = kde_.resample(count).T
    resample_data = np.column_stack((np.maximum(resample_data[:, 0], resample_data[:, 1]), np.minimum(resample_data[:, 0], resample_data[:, 1])))
    resample_data = resample_data.astype(int)

    oversize_data = resample_data[~((resample_data[:, 0] > 0) & (resample_data[:, 1] > 0)
                                    & (resample_data[:, 0] <= MATERIAL_SIZE[0]) &
                                    (resample_data[:, 1] <= MATERIAL_SIZE[1]))]

    oversize_data_clipped = np.clip(oversize_data, a_min=[20, 20], a_max=list(MATERIAL_SIZE))
    resample_data = resample_data[(resample_data[:, 0] > 0) & (resample_data[:, 1] > 0)
                                  & (resample_data[:, 0] <= MATERIAL_SIZE[0]) &
                                  (resample_data[:, 1] <= MATERIAL_SIZE[1])]
    resample_data = np.row_stack((resample_data, oversize_data_clipped))
    return_data: "np.ndarray" = np.column_stack((range(count), resample_data))
    return return_data


def test_rect():
    data = [
            [
                    [0, 0, 10, 10], [5, 5, 15, 15], [5, 5, 10, 10]
            ],
            [
                    [5, 0, 15, 10], [0, 5, 10, 15], [5, 5, 10, 10]
            ],
            [
                    [5, 5, 15, 15], [0, 0, 10, 10], [5, 5, 10, 10]
            ],
            [
                    [0, 5, 10, 15], [5, 0, 15, 10], [5, 5, 10, 10]
            ],
            [
                    [0, 5, 5, 10], [1, 0, 4, 6], [1, 5, 4, 6]
            ],
            [
                    [0, 0, 5, 5, ], [4, 1, 8, 4], [4, 1, 5, 4]
            ],
            [
                    [0, 0, 5, 5], [1, 4, 4, 6], [1, 4, 4, 5]
            ],
            [
                    [1, 0, 6, 5], [0, 1, 2, 4], [1, 1, 2, 4]
            ],
            [
                    [1, 0, 6, 5], [0, 1, 7, 4], [1, 1, 6, 4]
            ],
            [
                    [0, 1, 5, 6], [1, 0, 4, 7], [1, 1, 4, 6]
            ]
    ]
    for a, b, c in data:
        A = Rect(*a)
        B = Rect(*b)
        C = Rect(*c)
        D = A & B
        print(f"A={A},B={B},A&B={D}=={C},{D == C}", )




class EVAL:
    def __init__(self, algo_type, run_count, params):
        self.algo_type = algo_type
        self.run_count = run_count
        self.params = params

    def run(self, dataset):
        start = time()
        result = BinPacking2DAlgo.multi_run(dataset, MATERIAL_SIZE, run_count=self.run_count, algo_type=self.algo_type, parameter_input_array=self.params)
        print(result)
        print(time() - start)
        return result

    def run_single(self, dataset):
        start = time()
        result = BinPacking2DAlgo.single_run(dataset, MATERIAL_SIZE, algo_type=self.algo_type, parameter_input_array=self.params).get_avg_util_rate()
        print(f"{round(time() - start, 4)}s,{round(result * 100, 4)}%", end=", ")
        return result

    def run_time_cost(self, dataset):
        start = time()
        result = BinPacking2DAlgo.single_run(dataset, MATERIAL_SIZE, algo_type=self.algo_type, parameter_input_array=self.params).get_avg_util_rate()
        end = time()
        print(f"{round(end - start, 4)}s,{round(result * 100, 4)}%", end=", ")
        return end - start


params = {}




data_sets = {
        PRODUCTION_DATA1: 华为杯_data,
        PRODUCTION_DATA2: 外包_data,
        RANDOMGEN_DATA  : 随机_data,
        OPEN_ACCESS_DATA1:aa_data,
        OPEN_ACCESS_DATA2:cz_data

}
param_length={

}



class AlgoName:
    Dist_Skyline = "Dist_Skyline"
    Dist_MaxRect = "Dist_MaxRect"
    Dist_Shelf = "Dist_Shelf"
    Skyline = "Skyline"
    MaxRect = "MaxRect"

algo_types = [AlgoName.Skyline, AlgoName.MaxRect, AlgoName.Dist_Skyline, AlgoName.Dist_MaxRect, AlgoName.Dist_Shelf]
data_types = [NOISED, STANDARD]

for algo_name,gencount in [(AlgoName.Dist_MaxRect,500),(AlgoName.Dist_Skyline,500)]:
    params[algo_name] = {}
    for data_type in [STANDARD, NOISED]:
        params[algo_name][data_type] = {}
        for data_set_name in data_sets.keys():
            params[algo_name][data_type][data_set_name] = np.load(os.path.join(PARAM_PATH,f"{algo_name}_param_{data_type}_{data_set_name}_sample1000_gen{gencount}.npy"))


if __name__ == "__main__":
    for name,item in data_sets.items():
        print(item[item[:,2]>MATERIAL_SIZE[1]])
    pass
