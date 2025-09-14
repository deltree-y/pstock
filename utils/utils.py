# coding=utf-8
from collections import Counter
from datetime import datetime, timedelta
from math import ceil
from enum import Enum, auto
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import numpy as np
import logging, logging.config
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from numba import njit

def setup_logging():
    logging.basicConfig(
        #level=logging.DEBUG,
        level=logging.INFO,
        format='%(levelname)s %(message)s',
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

def feature_importance_analysis(ds, feature_names, n_features=25):
    """分析并选择最重要的特征"""
    feature_data = ds.raw_train_x[-len(ds.train_y):]  # 对齐数据
    target = ds.train_y[:, 0]  # 回归目标
    
    # 结合多种特征选择方法
    results = auto_select_features(feature_data, target, feature_names,
                              pearson_threshold=0.03, mi_threshold=0.01,
                              print_detail=True)
    
    # 使用随机森林获取特征重要性
    selected_rf, rf_scores = select_features_by_tree_importance(
        feature_data, target, feature_names,
        importance_threshold=0.01,
        print_detail=True
    )
    
    # 找出所有方法共同选出的特征
    common_features = set(results['pearson_selected']) & set(results['mi_selected']) & set(selected_rf)
    print(f"所有方法共同选出的特征 ({len(common_features)}):", common_features)
    
    # 找出至少被两种方法选出的特征
    features_selected_by_at_least_two = set()
    for f in set(feature_names):
        count = 0
        if f in results['pearson_selected']: count += 1
        if f in results['mi_selected']: count += 1
        if f in selected_rf: count += 1
        if count >= 2:
            features_selected_by_at_least_two.add(f)
    
    print(f"至少被两种方法选出的特征 ({len(features_selected_by_at_least_two)}):", 
          features_selected_by_at_least_two)
    
    # 使用这些特征的索引
    selected_indices = [i for i, name in enumerate(feature_names) 
                        if name in features_selected_by_at_least_two]
    
    return list(features_selected_by_at_least_two), selected_indices

def select_features_by_stat_corr(bin_labels, feature_data, feature_names, method='pearson', threshold=0.1):
    scores = []
    for i, fname in enumerate(feature_names):
        x = feature_data[:, i]
        if method == 'pearson':
            # 皮尔逊相关
            corr, _ = pearsonr(x, bin_labels)
            scores.append(abs(corr))
        elif method == 'mi':
            # 互信息
            mi = mutual_info_classif(x.reshape(-1, 1), bin_labels, discrete_features=False)
            scores.append(mi[0])
    scores = np.array(scores)
    selected_features = [feature_names[i] for i, s in enumerate(scores) if s > threshold]
    print("相关性得分:")
    for fname, score in zip(feature_names, scores):
        print(f"{fname}: {score:.3f}")
    print("筛选出的强相关特征:", selected_features)
    return selected_features

def auto_select_features(feature_data, target, feature_names,
                        pearson_threshold=0.15, mi_threshold=0.03,
                        print_detail=True):
    """
    自动筛选与回归目标相关性强的特征
    :param feature_data: shape [n_samples, n_features]
    :param target: shape [n_samples]
    :param feature_names: list of feature names
    :param pearson_threshold: 皮尔逊相关系数筛选阈值（绝对值）
    :param mi_threshold: 互信息筛选阈值
    :param print_detail: 是否打印全部特征得分
    :return: dict, 包含皮尔逊强相关、互信息强相关特征列表
    """
    pearson_scores = []
    for i in range(feature_data.shape[1]):
        corr, _ = pearsonr(feature_data[:, i], target)
        pearson_scores.append(abs(corr))
    mi_scores = mutual_info_regression(feature_data, target)
    
    pearson_selected = [feature_names[i] for i, s in enumerate(pearson_scores) if s > pearson_threshold]
    mi_selected = [feature_names[i] for i, s in enumerate(mi_scores) if s > mi_threshold]

    if print_detail:
        print("=== 皮尔逊相关系数 ===")
        for fname, score in zip(feature_names, pearson_scores):
            print(f"{fname}: {score:.3f}")
        print("皮尔逊强相关特征:", pearson_selected)
        print("\n=== 互信息 ===")
        for fname, score in zip(feature_names, mi_scores):
            print(f"{fname}: {score:.3f}")
        print("互信息强相关特征:", mi_selected)
    return {
        "pearson_selected": pearson_selected,
        "mi_selected": mi_selected,
        "pearson_scores": pearson_scores,
        "mi_scores": mi_scores
    }

def select_features_by_tree_importance(feature_data, target, feature_names, importance_threshold=0.01, print_detail=True):
    """
    用随机森林回归筛选与目标相关性强的特征
    :param feature_data: shape [n_samples, n_features]
    :param target: shape [n_samples]
    :param feature_names: list of feature names
    :param importance_threshold: 特征重要性阈值
    :param print_detail: 是否打印全部特征得分
    :return: list, 强相关特征名
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feature_data, target)
    importances = rf.feature_importances_
    selected = [feature_names[i] for i, imp in enumerate(importances) if imp > importance_threshold]
    if print_detail:
        print("=== 随机森林特征重要性 ===")
        for fname, score in zip(feature_names, importances):
            print(f"{fname}: {score:.4f}")
        print("随机森林强相关特征:", selected)
    return selected, importances

def plot_regression_result(y_true, y_pred, title="回归预测结果", save_path=None):
    """
    可视化回归预测结果
    :param y_true: 测试集真实标签 (一维数组)
    :param y_pred: 测试集预测值 (一维数组)
    :param title: 图标题
    :param save_path: 若指定则保存图片
    """
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
    """
    可视化预测误差分布
    :param y_true: 测试集真实标签 (一维数组)
    :param y_pred: 测试集预测值 (一维数组)
    :param title: 图标题
    :param save_path: 若指定则保存图片
    """
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
    logging.info(f"count of {label}:")
    counter, total = Counter(lst), len(lst)
    for num, count in counter.items():
        percent = count / total
        logging.info(f"{num}: {percent:.2%}")    

class SuperList(list):
    def append(self, item):
        super().append(item)
        return self


class StockType(Enum):
    PRIMARY = auto()
    RELATED = auto()
    INDEX = auto()