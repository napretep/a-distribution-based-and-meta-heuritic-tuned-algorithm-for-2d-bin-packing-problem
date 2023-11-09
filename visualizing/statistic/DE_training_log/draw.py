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
    data1 = "Dist2_production_data1__500_traning_log__.npy"
    data2 = "Dist2_production_data2__500_traning_log__.npy"
    data3 = "Dist2_random_data__500_traning_log__.npy"
    data1_random = "Dist2_production_data1_random_(0, 0.3)_1500_traning_log__.npy" #"Dist2_production_data1_random_(0, 0.3)_500_traning_log__.npy"
    data2_random = "Dist2_production_data2_random_(0, 0.3)_500_traning_log__.npy"
    dist3_random = "Dist2_random_data_random_(0, 0.3)_500_traning_log__.npy"

    # Load data
    data1 = np.load(data1)[:,1]
    data2 = np.load(data2)[:,1]
    data3 = np.load(data3)[:,1]
    data1_random = np.load(data1_random)[:,1]
    data2_random = np.load(data2_random)[:,1]
    dist3_random = np.load(dist3_random)[:,1]
    print(data1.shape)
    # Create a new figure for the first set of data
    # Calculate moving averages
    # Calculate spline
    # Calculate moving averages
    data1_ma = pd.Series(data1).rolling(window=10).mean().dropna()
    data2_ma = pd.Series(data2).rolling(window=10).mean().dropna()
    data3_ma = pd.Series(data3).rolling(window=10).mean().dropna()
    data1_random_ma = pd.Series(data1_random).rolling(window=10).mean().dropna()
    data2_random_ma = pd.Series(data2_random).rolling(window=10).mean().dropna()
    dist3_random_ma = pd.Series(dist3_random).rolling(window=10).mean().dropna()


    x1 = np.arange(len(data1_ma))
    slope1, intercept1, _, _, _ = linregress(x1, data1_ma)
    x2 = np.arange(len(data2_ma))
    slope2, intercept2, _, _, _ = linregress(x2, data2_ma)
    x3 = np.arange(len(data3_ma))
    slope3, intercept3, _, _, _ = linregress(x3, data3_ma)


    x4 = np.arange(len(data1_random_ma))
    slope4, intercept4, _, _, _ = linregress(x4, data1_random_ma)
    x5 = np.arange(len(data2_random_ma))
    slope5, intercept5, _, _, _ = linregress(x5, data2_random_ma)
    x6 = np.arange(len(dist3_random_ma))
    slope6, intercept6, _, _, _ = linregress(x6, dist3_random_ma)
    # Create a new figure for the first set of data
    plt.figure()
    plt.plot(data1_ma, label="production_data1")
    plt.plot(data2_ma, label="production_data2")
    plt.plot(data3_ma, label="random_data")
    plt.plot(x1, intercept1 + slope1 * x1, 'r', )
    plt.plot(x2, intercept2+ slope2 * x2, 'g',  )
    plt.plot(x3, intercept3+ slope3 * x3, 'b',  )
    plt.legend()
    plt.title("multi-eval training log (Moving Average & linearRegression)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig("multi-eval_training_log_(Moving_Average_and_linearRegression).png")
    # Create a new figure for the second set of data
    plt.figure()
    plt.plot(data1_random_ma, label="production_data1 with noise")
    plt.plot(data2_random_ma, label="production_data2 with noise")
    plt.plot(dist3_random_ma, label="random_data with noise")
    plt.plot(x4, intercept4 + slope4 * x4, 'r',  )
    plt.plot(x5, intercept5 + slope5 * x5, 'g',  )
    plt.plot(x6, intercept6 + slope6 * x6, 'b',  )
    plt.legend()
    plt.title("multi-eval training log (Moving Average & linearRegression & with noise)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig("multi-eval_training_log_(Moving_Average_and_linearRegression_with_noise).png")
    plt.show()

if __name__ == "__main__":
    run()
    pass