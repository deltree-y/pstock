# coding=utf-8
from collections import Counter
from enum import Enum, auto
import numpy as np
import pandas as pd
import logging
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
        print(f"[{num}]: {percent:.1%},", end='')
    print()

def print_nan_inf_info(arr, name):
    #np.set_printoptions(threshold=np.inf)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 100)
    print(f"检查 {name}:")
    nan_rows = np.where(np.isnan(arr).any(axis=1))[0]
    inf_rows = np.where(np.isinf(arr).any(axis=1))[0]
    if len(nan_rows) > 0:
        print(f"{name} 存在 NaN 的行索引: {nan_rows}")
    if len(inf_rows) > 0:
        print(f"{name} 存在 Inf 的行索引: {inf_rows}")    
    #if len(nan_rows) > 0:
    #    tx_nan_last_step = tx[nan_rows, -1, :]   # [nan样本数, 特征数]
    #    print(pd.DataFrame(tx_nan_last_step))        
        #df = pd.DataFrame(arr[nan_rows])
        #print(f"{name} NaN行详细内容:\n{df.describe(include='all')}\n{df.head()}")

class SuperList(list):
    def append(self, item):
        super().append(item)
        return self
    
class StrategyType(Enum):
    BUY =  "+"
    SELL = "-"
    HOLD = "o"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

class StockType(Enum):
    PRIMARY = auto()
    RELATED = auto()
    INDEX = auto()

class ModelType(Enum):
    RESIDUAL_LSTM = 'ResLSTM'
    RESIDUAL_TCN = 'ResTCN'
    TRANSFORMER = 'Transformer'
    CONV1D = 'Conv1D'
    MINI = 'Mini'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class FeatureType(Enum):
    ALL = 'all_features'

    CLASSIFY_F50 = 'classify_features_50'
    CLASSIFY_F30 = 'classify_features_30'

    T1L05_F35 = 't1l05_features_35'
    T1L05_F55 = 't1l05_features_55'

    T1L08_F30 = 't1l08_features_30'

    T1L10_F15 = 't1l10_features_15'
    T1L10_F35 = 't1l10_features_35'
    T1L10_F55 = 't1l10_features_55'

    T1H05_F55 = 't1h05_features_55'

    T1H08_F18 = 't1h08_features_18' 

    T1H10_F35 = 't1h10_features_35'
    T1H10_F55 = 't1h10_features_55'
    T1H10_F75 = 't1h10_features_75'

    T1H15_F35 = 't1h10_features_35' #TODO: 是否应为t1h15_features_35？
    T1H15_F55 = 't1h10_features_55' #TODO: 是否应为t1h15_features_55？
    T1H15_F75 = 't1h10_features_75' #TODO: 是否应为t1h15_features_75？

    T1L15_F35 = 't1l15_features_35'
    T1L15_F55 = 't1l15_features_55'
    T1L15_F75 = 't1l15_features_75'

    T2H05_F55 = 't2h05_features_55' #TODO:待添加真实特征

    T2H08_F55 = 't2h08_features_55' #TODO:待添加真实特征

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
    BINARY_T1_L08 = ("BINARY_T1_L08", -0.8, "T1L")
    BINARY_T1_L10 = ("BINARY_T1_L10", -1.0, "T1L")
    BINARY_T1_L15 = ("BINARY_T1_L15", -1.5, "T1L") 

    BINARY_T1_H05 = ("BINARY_T1_H05", 0.5, "T1H")
    BINARY_T1_H08 = ("BINARY_T1_H08", 0.8, "T1H")
    BINARY_T1_H10 = ("BINARY_T1_H10", 1.0, "T1H")
    BINARY_T1_H15 = ("BINARY_T1_H15", 1.5, "T1H")

    BINARY_T2_L05 = ("BINARY_T2_L05", -0.5, "T2L")
    BINARY_T2_L08 = ("BINARY_T2_L08", -0.8, "T2L")
    BINARY_T2_L10 = ("BINARY_T2_L10", -1.0, "T2L")
    BINARY_T2_L15 = ("BINARY_T2_L15", -1.5, "T2L")

    BINARY_T2_H05 = ("BINARY_T2_H05", 0.5, "T2H")
    BINARY_T2_H08 = ("BINARY_T2_H08", 0.8, "T2H")
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

