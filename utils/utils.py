# coding=utf-8
from datetime import datetime, timedelta
from math import ceil
from enum import Enum, auto
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import numpy as np
import logging, logging.config

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
    
class SuperList(list):
    def append(self, item):
        super().append(item)
        return self


class StockType(Enum):
    PRIMARY = auto()
    RELATED = auto()
    INDEX = auto()