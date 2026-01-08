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
    print(f"{label} 数据分布统计(min: {lst.min()}, max: {lst.max()})/total:{len(lst)}:", end="")
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
    CONV2D = 'Conv2D'
    MINI = 'Mini'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value


class FeatureType(Enum):
    #以auto()方式生成枚举值时，其取值为其名称的小写形式(如T1l='t1l')
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()   # 或 name 本身、不用lower()
        
    ALL = 'all_features'

    BINARY_T1L03_F55 = auto()
    BINARY_T1L04_F55 = auto()
    BINARY_T1L05_F35 = auto()
    BINARY_T1L05_F55 = auto()
    BINARY_T1L06_F55 = auto()
    BINARY_T1L07_F55 = auto()
    BINARY_T1L08_F55 = auto()
    BINARY_T1L08_F30 = auto()
    BINARY_T1L10_F15, BINARY_T1L10_F35, BINARY_T1L10_F55 = auto(), auto(), auto()
    BINARY_T1L15_F35, BINARY_T1L15_F55, BINARY_T1L15_F75 = auto(), auto(), auto()
    BINARY_T1L20_F55 = auto()
    BINARY_T1L30_F55 = auto()

    BINARY_T1H05_F55 = auto()
    BINARY_T1H08_F18 = auto()
    BINARY_T1H10_F35, BINARY_T1H10_F55, BINARY_T1H10_F75 = auto(), auto(), auto()
    BINARY_T1H15_F35, BINARY_T1H15_F55, BINARY_T1H15_F75 = auto(), auto(), auto()

    BINARY_T2H03_F55 = auto() #TODO:待添加真实特征
    BINARY_T2H04_F55 = auto() #TODO:待添加真实特征
    BINARY_T2H05_F55 = auto() #TODO:待添加真实特征
    BINARY_T2H06_F55 = auto() #TODO:待添加真实特征
    BINARY_T2H07_F55 = auto() #TODO:待添加真实特征
    BINARY_T2H08_F55 = auto() #TODO:待添加真实特征
    BINARY_T2H10_F10, BINARY_T2H10_F25, BINARY_T2H10_F35, BINARY_T2H10_F45, BINARY_T2H10_F55 = auto(), auto(), auto(), auto(), auto()
    
    REGRESS_T1L_F55, REGRESS_T1L_F50 = auto(), auto()
    REGRESS_T1H_F72, REGRESS_T1H_F55, REGRESS_T1H_F50 = auto(), auto(), auto()
    REGRESS_T2H_F55, REGRESS_T2H_F50, REGRESS_T2H_F30, REGRESS_T2H_F20 = auto(), auto(), auto(), auto()

    CLASSIFY_F50 = 'classify_features_50'
    CLASSIFY_F30 = 'classify_features_30'


    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    @property
    def short_name(self):
        if self == FeatureType.ALL:
            return "ALL"
        else:   #返回"xxxxx_Fxx"字样
            s = self.name
            if s[:3] == "BIN":
                return s[s.rfind('_')-2:s.rfind('_')] + s[s.rfind('_')+1:]
            elif s[:3] == "REG":
                return s.split("_", 2)[2] if s.count("_") >= 2 else s
            elif s[:5] == "CLASS":
                raise "需要增加对应处理"#TODO
    
class PredictType(Enum):
    BINARY_T1L03 = ("BINARY_T1_L03", -0.3, "BIN_T1L")
    BINARY_T1L04 = ("BINARY_T1_L04", -0.4, "BIN_T1L")
    BINARY_T1L05 = ("BINARY_T1_L05", -0.5, "BIN_T1L")
    BINARY_T1L06 = ("BINARY_T1_L06", -0.6, "BIN_T1L")
    BINARY_T1L07 = ("BINARY_T1_L07", -0.7, "BIN_T1L")
    BINARY_T1L08 = ("BINARY_T1_L08", -0.8, "BIN_T1L")
    BINARY_T1L10 = ("BINARY_T1_L10", -1.0, "BIN_T1L")
    BINARY_T1L15 = ("BINARY_T1_L15", -1.5, "BIN_T1L") 
    BINARY_T1L20 = ("BINARY_T1_L20", -2.0, "BIN_T1L") 
    BINARY_T1L30 = ("BINARY_T1_L30", -3.0, "BIN_T1L") 

    BINARY_T1H05 = ("BINARY_T1_H05", 0.5, "BIN_T1H")
    BINARY_T1H06 = ("BINARY_T1_H06", 0.6, "BIN_T1H")
    BINARY_T1H07 = ("BINARY_T1_H07", 0.7, "BIN_T1H")
    BINARY_T1H08 = ("BINARY_T1_H08", 0.8, "BIN_T1H")
    BINARY_T1H10 = ("BINARY_T1_H10", 1.0, "BIN_T1H")
    BINARY_T1H15 = ("BINARY_T1_H15", 1.5, "BIN_T1H")

    BINARY_T2L05 = ("BINARY_T2_L05", -0.5, "BIN_T2L")
    BINARY_T2L08 = ("BINARY_T2_L08", -0.8, "BIN_T2L")
    BINARY_T2L10 = ("BINARY_T2_L10", -1.0, "BIN_T2L")
    BINARY_T2L15 = ("BINARY_T2_L15", -1.5, "BIN_T2L")

    BINARY_T2H03 = ("BINARY_T2_H03", 0.3, "BIN_T2H")
    BINARY_T2H04 = ("BINARY_T2_H04", 0.4, "BIN_T2H")
    BINARY_T2H05 = ("BINARY_T2_H05", 0.5, "BIN_T2H")
    BINARY_T2H06 = ("BINARY_T2_H06", 0.6, "BIN_T2H")
    BINARY_T2H07 = ("BINARY_T2_H07", 0.7, "BIN_T2H")
    BINARY_T2H08 = ("BINARY_T2_H08", 0.8, "BIN_T2H")
    BINARY_T2H10 = ("BINARY_T2_H10", 1.0, "BIN_T2H")
    BINARY_T2H15 = ("BINARY_T2_H15", 1.5, "BIN_T2H")

    REGRESS_T1L =  ("REGRESS_T1L",  1100.0, "REG_T1L")
    REGRESS_T1H =  ("REGRESS_T1H",  1200.0, "REG_T1H")
    REGRESS_T2H =  ("REGRESS_T2H",  1300.0, "REG_T2H")

    CLASSIFY = ("CLASSIFY", 100.0, "CLASSIFY")

    def __str__(self):
        return self.value[2]
    
    def __repr__(self):
        return self.value[2]
    
    def get_type_from_feature_type(ft:FeatureType):
        name = ft.name
        if name.startswith("BINARY"):
            if "T1L" in name:
                return PredictType["BINARY_T1L" + name.split("_")[1][-2:]]
            elif "T1H" in name:
                return PredictType["BINARY_T1H" + name.split("_")[1][-2:]]
            elif "T2L" in name:
                return PredictType["BINARY_T2L" + name.split("_")[1][-2:]]
            elif "T2H" in name:
                return PredictType["BINARY_T2H" + name.split("_")[1][-2:]]
            else:
                raise ValueError(f"无法从特征类型 {ft} 推断出对应的预测类型")
        elif name.startswith("REGRESS"):
            if "T1L" in name:
                return PredictType.REGRESS_T1L
            elif "T1H" in name:
                return PredictType.REGRESS_T1H
            elif "T2H" in name:
                return PredictType.REGRESS_T2H
            else:
                raise ValueError(f"无法从特征类型 {ft} 推断出对应的预测类型")
        elif name.startswith("CLASSIFY"):
            return PredictType.CLASSIFY
        else:
            raise ValueError(f"无法从特征类型 {ft} 推断出对应的预测类型")

    @property
    def val(self):
        return self.value[1]
    
    @property
    def label(self):
        return self.value[2]

    def is_binary(self):
        return self.value[0][:6] == "BINARY"

    def is_classify(self):
        return self.value[0][:8] == "CLASSIFY"
    
    def is_regression(self):
        return self.value[0][:7] == "REGRESS"

    def is_t1_low(self):
        return self.value[2][-3:] == "T1L"
    
    def is_t1_high(self):
        return self.value[2][-3:] == "T1H"
    
    def is_t2_low(self):
        return self.value[2][-3:] == "T2L"
    
    def is_t2_high(self):
        return self.value[2][-3:] == "T2H"
    
    def is_low(self):
        return self.value[2][-1] == "L"

    def is_high(self):
        return self.value[2][-1] == "H"

