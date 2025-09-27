# coding=utf-8
from collections import Counter
from datetime import datetime, timedelta
from math import ceil
from enum import Enum, auto
import numpy as np
import logging, logging.config
import matplotlib.pyplot as plt
from numba import njit

def setup_logging():
    logging.basicConfig(
        #level=logging.DEBUG,
        level=logging.INFO,
        format='%(message)s',
        #format='%(levelname)s %(message)s',
        #format='%(asctime)s %(levelname)s %(message)s',
        #format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        #format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d %(funcName)s] %(message)s',
        datefmt='%H:%M:%S'
    )

def get_predict_score(pred, real):
    score = 0
    if pred == real:    #完全预测正确
        score = 1       #则可得满分
    #elif abs(real-pred) == 1: #预测范围空间差距1档
    #    score = 0.5           #预测范围空间差距1档，得0.5分

    #elif (real<=4 and pred<=4) or (real>=5 and pred>=5):    #预测涨跌正确
    #    score = score + 0.5
    #    if abs(real-pred) == 1:     #预测范围空间差距1档
    #        score = score + 0.25
    #    elif abs(real-pred) == 2:   #预测范围空间差距2档
    #        score = score + 0.1
    #    else:
    #        pass
    else:
        pass

    return score

def get_minor_one_day(input_date):
    ret_date =  get_datetime_date(input_date) - timedelta(days=1)
    ret_date_in_string = get_string_date(ret_date)
    #print("INFO: get_minor_one_day() - in date is :<%s>, ret date is :<%s>"%(input_date, ret_date_in_string))
    return ret_date_in_string

def get_datetime_date(input_date):
    return datetime.strptime(input_date,"%Y%m%d")

def day_plus_minor(input_date, dd):
    id = get_datetime_date(input_date)
    return (id+timedelta(days=dd)).strftime("%Y%m%d")

def get_string_date(input_datetime):
    return input_datetime.strftime("%Y%m%d")

def get_mind_value(value, base_value):
    if value <= base_value:
        if ceil(value) - value < 0.02:
            return ceil(value) + 0.03
    else:
        if value - int(value) < 0.02:
            return int(value) - 0.03
    return value



def plot_regression_result(y_true, y_pred, title="回归预测结果", save_path=None):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="real", marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label="pred", marker='x', linestyle='--', alpha=0.7)
    plt.ylim(-8, 2)
    plt.title(title)
    plt.xlabel("sn")
    plt.ylabel("chg_pct(%)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_error_distribution(y_true, y_pred, title="mae/rmse distribution", save_path=None):
    errors = y_pred - y_true
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("mae(pred-real,%)")
    plt.ylabel("sample count")
    plt.xlim(-5, 5)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

@njit
def rolling_skew(arr, window):
    n = arr.shape[0]
    result = np.empty(n)
    result[:] = np.nan
    for i in range(window - 1, n):
        x = arr[i - window + 1:i + 1]
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            result[i] = 0.0  # or np.nan
        else:
            result[i] = np.mean(((x - mean) / std) ** 3)
    return result

@njit
def rolling_kurtosis(arr, window):
    n = arr.shape[0]
    result = np.empty(n)
    result[:] = np.nan
    for i in range(window - 1, n):
        x = arr[i - window + 1:i + 1]
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            result[i] = -3.0  # or np.nan
        else:
            result[i] = np.mean(((x - mean) / std) ** 4) - 3
    return result

def print_ratio(lst, label=""):
    print(f"count of {label}(min: {lst.min()}, max: {lst.max()}):")
    counter, total = Counter(lst), len(lst)
    for num, count in sorted(counter.items()):
        percent = count / total
        print(f"[{num}: {percent:.1%}]", end=' ')
    print()

class SuperList(list):
    def append(self, item):
        super().append(item)
        return self


class StockType(Enum):
    PRIMARY = auto()
    RELATED = auto()
    INDEX = auto()

class PredictType(Enum):
    BINARY_T1_L05 = ("BINARY_T1_L05", -0.5)
    BINARY_T1_L10 = ("BINARY_T1_L10", -1.0)
    BINARY_T1_L15 = ("BINARY_T1_L15", -1.5)

    BINARY_T1_H05 = ("BINARY_T1_H05", 0.5)
    BINARY_T1_H10 = ("BINARY_T1_H10", 1.0)
    BINARY_T1_H15 = ("BINARY_T1_H15", 1.5)

    BINARY_T2_L05 = ("BINARY_T2_L05", -0.5)
    BINARY_T2_L10 = ("BINARY_T2_L10", -1.0)
    BINARY_T2_L15 = ("BINARY_T2_L15", -1.5)

    BINARY_T2_H05 = ("BINARY_T2_H05", 0.5)
    BINARY_T2_H10 = ("BINARY_T2_H10", 1.0)
    BINARY_T2_H15 = ("BINARY_T2_H15", 1.5)


    CLASSIFY = ("classify", 100.0)
    REGRESS = ("regress", 1000.0)

    def is_binary(self):
        return self in [
            PredictType.BINARY_T1_L05,
            PredictType.BINARY_T1_L10,
            PredictType.BINARY_T1_L15,
            PredictType.BINARY_T1_H05,
            PredictType.BINARY_T1_H10,
            PredictType.BINARY_T1_H15,
            PredictType.BINARY_T2_L05,
            PredictType.BINARY_T2_L10,
            PredictType.BINARY_T2_L15,
            PredictType.BINARY_T2_H05,
            PredictType.BINARY_T2_H10,
            PredictType.BINARY_T2_H15
        ]
    
    def is_binary_t1_low(self):
        return self in [
            PredictType.BINARY_T1_L05,
            PredictType.BINARY_T1_L10,
            PredictType.BINARY_T1_L15
        ]
    
    def is_binary_t1_high(self):
        return self in [
            PredictType.BINARY_T1_H05,
            PredictType.BINARY_T1_H10,
            PredictType.BINARY_T1_H15
        ]
    
    def is_binary_t2_low(self):
        return self in [
            PredictType.BINARY_T2_L05,
            PredictType.BINARY_T2_L10,
            PredictType.BINARY_T2_L15
        ]
    
    def is_binary_t2_high(self):
        return self in [
            PredictType.BINARY_T2_H05,
            PredictType.BINARY_T2_H10,
            PredictType.BINARY_T2_H15
        ]

    def is_classify(self):
        return self in [
            PredictType.CLASSIFY
            ]
    
    def is_regress(self):
        return self in [
            PredictType.REGRESS
            ]

    @property
    def val(self):
        """返回阈值(float), 无则None"""
        v = self.value[1]
        return v if isinstance(v, float) else None