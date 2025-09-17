import warnings
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

def binary_class_feature_selection(x_data, y_label, feature_names, class_a=0, class_b=5):
    # 只保留0和5
    mask = (y_label == class_a) | (y_label == class_b)
    x_sub = x_data[mask]
    y_sub = (y_label[mask] == class_b).astype(int)  # 二分类

    mi_scores = mutual_info_classif(x_sub, y_sub)
    pearson_scores = [abs(pearsonr(x_sub[:,i], y_sub)[0]) for i in range(x_sub.shape[1])]
    print("互信息得分(前10):", sorted(zip(feature_names, mi_scores), key=lambda x:-x[1])[:10])
    print("皮尔逊得分(前10):", sorted(zip(feature_names, pearson_scores), key=lambda x:-x[1])[:10])
    return mi_scores, pearson_scores


def feature_importance_analysis(ds, feature_names, pearson_threshold=0.03, mi_threshold=0.01, importance_threshold=0.01, n_features=25):
    """分析并选择最重要的特征"""
    feature_data = ds.raw_train_x[-len(ds.train_y):]  # 对齐数据
    target = ds.train_y[:, 0]  # 回归目标
    
    # 结合多种特征选择方法
    results = auto_select_features(feature_data, target, feature_names,
                              pearson_threshold=pearson_threshold, mi_threshold=mi_threshold,
                              print_detail=True)
    
    # 使用随机森林获取特征重要性
    selected_rf, rf_scores = select_features_by_tree_importance(
        feature_data, target, feature_names,
        importance_threshold=importance_threshold,
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = pearsonr(feature_data[:, i], target)
        pearson_scores.append(abs(corr))
    mi_scores = mutual_info_regression(feature_data, target)
    
    pearson_selected = [feature_names[i] for i, s in enumerate(pearson_scores) if s > pearson_threshold]
    mi_selected = [feature_names[i] for i, s in enumerate(mi_scores) if s > mi_threshold]

    if print_detail:
        print("=== 皮尔逊相关系数 ===")
        for fname, score in zip(feature_names, pearson_scores):
            print(f"{fname}: {score:.3f}")
        print(f"皮尔逊强相关特征({len(pearson_selected)}):", pearson_selected)
        print("\n=== 互信息 ===")
        for fname, score in zip(feature_names, mi_scores):
            print(f"{fname}: {score:.3f}")
        print(f"互信息强相关特征({len(mi_selected)}):", mi_selected)
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