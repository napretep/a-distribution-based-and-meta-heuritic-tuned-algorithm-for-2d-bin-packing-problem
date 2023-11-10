# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'draw.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/9 22:44'
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
from scipy.stats import linregress

def run():

    determ_data = [
            "Dist2_production_data1__1500_traning_log__1699576099.npy",
            "Dist2_production_data2__500_traning_log__.npy",
            "Dist2_production_data2__1500_traning_log__1699578250.npy" ,
            "Dist2_production_data2__500_traning_log__.npy",
            "Dist2_random_data__500_traning_log__.npy"
    ]

    noised_data =[
            "Dist2_production_data1_random_(0, 0.3)_1500_traning_log__.npy",
            "Dist2_production_data1_random_(0, 0.3)_500_traning_log__.npy",
            "Dist2_production_data2_random_(0, 0.3)_500_traning_log__.npy",
            "Dist2_random_data_random_(0, 0.3)_500_traning_log__.npy"
    ]
    plt.figure()
    for data_path in determ_data:
        data = np.load(data_path)[:250, 1]
        data_ma = pd.Series(data).rolling(window=10).mean().dropna()
        x = np.arange(len(data_ma))
        slope, intercept, _, _, _ = linregress(x, data_ma)
        plt.plot(data_ma, label=data_path)
        plt.plot(x, intercept + slope * x)
    plt.legend()
    plt.title("multi-eval training log (Moving Average & linearRegression)")
    plt.xlabel("generations")
    plt.ylabel("Value")
    plt.savefig("multi-eval_training_log_(Moving_Average_and_linearRegression).png")
    plt.show()
    plt.figure()
    for data_path in noised_data:
        data = np.load(data_path)[:250, 1]
        data_ma = pd.Series(data).rolling(window=10).mean().dropna()
        x = np.arange(len(data_ma))
        slope, intercept, _, _, _ = linregress(x, data_ma)
        plt.plot(data_ma, label=data_path)
        plt.plot(x, intercept + slope * x)
    plt.legend()
    plt.title("multi-eval training log (Moving Average & linearRegression & with noise)")
    plt.xlabel("generations")
    plt.ylabel("Value")
    plt.savefig("multi-eval_training_log_(Moving_Average_and_linearRegression_with_noise).png")

    plt.show()

if __name__ == "__main__":
    run()
    pass