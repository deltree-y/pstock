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
    print(f"{label} 数据分布统计(min: {lst.min()}, max: {lst.max()}):", end="")
    counter, total = Counter(lst), len(lst)
    for num, count in sorted(counter.items()):
        percent = count / total
        print(f" [{num}]: {percent:.1%}, ", end=' ')
    print()

class SuperList(list):
    def append(self, item):
        super().append(item)
        return self


class StockType(Enum):
    PRIMARY = auto()
    RELATED = auto()
    INDEX = auto()

class ModelType(Enum):
    RESIDUAL_LSTM = 'ResLSTM'
    RESIDUAL_TCN = 'ResTCN'
    TRANSFORMER = 'Transformer'
    MINI = 'Mini'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class FeatureType(Enum):
    ALL = 'all_features'

    T1L05_F35 = 't1l05_features_35'
    T1L05_F55 = 't1l05_features_55'

    T1L10_F25 = 't1l10_features_25'
    T1L10_F35 = 't1l10_features_35'
    T1L10_F45 = 't1l10_features_45'
    T1L10_F55 = 't1l10_features_55'

    T1L15_F35 = 't1l15_features_35'
    T1L15_F55 = 't1l15_features_55'
    T1L15_F75 = 't1l15_features_75'

    T2H10_F25 = 't2h10_features_25'
    T2H10_F35 = 't2h10_features_35'
    T2H10_F45 = 't2h10_features_45'
    T2H10_F55 = 't2h10_features_55'

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    @property
    def short_name(self):
        if self == FeatureType.ALL:
            return "ALL"
        else:   #返回"xx_Fxx"字样
            s = self.name
            return s[s.rfind('_')-2:s.rfind('_')] + s[s.rfind('_')+1:]
    
class PredictType(Enum):
    BINARY_T1_L05 = ("BINARY_T1_L05", -0.5, "T1L")
    BINARY_T1_L10 = ("BINARY_T1_L10", -1.0, "T1L")
    BINARY_T1_L15 = ("BINARY_T1_L15", -1.5, "T1L") 

    BINARY_T1_H05 = ("BINARY_T1_H05", 0.5, "T1H")
    BINARY_T1_H10 = ("BINARY_T1_H10", 1.0, "T1H")
    BINARY_T1_H15 = ("BINARY_T1_H15", 1.5, "T1H")

    BINARY_T2_L05 = ("BINARY_T2_L05", -0.5, "T2L")
    BINARY_T2_L10 = ("BINARY_T2_L10", -1.0, "T2L")
    BINARY_T2_L15 = ("BINARY_T2_L15", -1.5, "T2L")

    BINARY_T2_H05 = ("BINARY_T2_H05", 0.5, "T2H")
    BINARY_T2_H10 = ("BINARY_T2_H10", 1.0, "T2H")
    BINARY_T2_H15 = ("BINARY_T2_H15", 1.5, "T2H")

    CLASSIFY = ("CLASSIFY", 100.0, "CLASSIFY")
    REGRESS =  ("REGRESS",  1000.0, "REGRESS")

    def __str__(self):
        return self.value[2]
    
    def __repr__(self):
        return self.value[2]

    @property
    def val(self):
        return self.value[1]
    
    @property
    def label(self):
        return self.value[2]

    def is_binary(self):
        return self.value[0][:6] == "BINARY"
    
    def is_binary_t1_low(self):
        return self.label == "T1L"
    
    def is_binary_t1_high(self):
        return self.label == "T1H"
    
    def is_binary_t2_low(self):
        return self.label == "T2L"
    
    def is_binary_t2_high(self):
        return self.label == "T2H"
    
    def is_classify(self):
        return self.label == "CLASSIFY"

    def is_regress(self):
        return self.label == "REGRESS"

