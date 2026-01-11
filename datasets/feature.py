import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import pearsonr
from datasets.dataset import StockDataset

def binary_class_feature_selection(x_train, y_train, feature_names):
    mask = np.isin(y_train, [0, 5])
    x_bin = x_train[mask]
    y_bin = y_train[mask]
    y_bin = (y_bin == 5).astype(int)

    print("x_bin shape:", x_bin.shape)
    print("y_bin shape:", y_bin.shape)
    print("feature_names:", len(feature_names))

    # 防止特征数量不一致
    if len(feature_names) != x_bin.shape[1]:
        print("Warning: feature_names与x_bin特征数不一致，自动裁剪")
        feature_names = feature_names[:x_bin.shape[1]]

    pearson_scores = [abs(pearsonr(x_bin[:, i], y_bin)[0]) for i in range(x_bin.shape[1])]
    mi_scores = mutual_info_classif(x_bin, y_bin, discrete_features=False, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(x_bin, y_bin)
    rf_scores = rf.feature_importances_

    print("feature_names:", len(feature_names))
    print("pearson_scores:", len(pearson_scores))
    print("mi_scores:", len(mi_scores))
    print("rf_scores:", len(rf_scores))

    df = pd.DataFrame({
        'feature': feature_names,
        'pearson': pearson_scores,
        'mi': mi_scores,
        'rf': rf_scores
    })
    df['score'] = df[['pearson', 'mi', 'rf']].mean(axis=1)
    df.sort_values('score', ascending=False, inplace=True)
    return df


def multiclass_feature_selection(x_train, y_train, feature_names):
    """
    针对全部类别做特征区分度分析
    返回：区分度排名表DataFrame
    """
    # 皮尔逊相关性（与类别标签相关性，类别标签不适合皮尔逊但可以粗略参考）
    pearson_scores = [abs(pearsonr(x_train[:,i], y_train)[0]) for i in range(x_train.shape[1])]
    # 互信息
    mi_scores = mutual_info_classif(x_train, y_train, discrete_features=False, random_state=42)
    # 随机森林重要性
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(x_train, y_train)
    rf_scores = rf.feature_importances_
    
    df = pd.DataFrame({
        'feature': feature_names,
        'pearson': pearson_scores,
        'mi': mi_scores,
        'rf': rf_scores
    })
    df['score'] = df[['pearson', 'mi', 'rf']].mean(axis=1)
    df.sort_values('score', ascending=False, inplace=True)
    return df

def select_joint_features(df_bin, df_multi, top_k=15, alpha=0.5):
    """
    从二分类（0/5）和多分类分析结果中，选出兼顾两者的前top_k特征
    alpha: 二分类权重（0~1），越大偏重极端类别判别
    """
    # 按名字合并
    df = df_bin[['feature','score']].merge(df_multi[['feature','score']], on='feature', suffixes=('_bin','_multi'))
    # 综合分数
    df['joint_score'] = df['score_bin']*alpha + df['score_multi']*(1-alpha)
    df.sort_values('joint_score', ascending=False, inplace=True)
    return df['feature'].head(top_k).tolist()


def feature_importance_analysis(ds, feature_names, pearson_threshold=0.03, mi_threshold=0.01, importance_threshold=0.01, n_features=55):
    """分析并选择最重要的特征"""
    feature_data = ds.raw_train_x[-len(ds.train_y):]  # 对齐数据
    target = ds.train_y[:, 0]  # 回归目标
    # 修复：填充所有nan/inf
    feature_data = np.nan_to_num(feature_data, nan=0, posinf=0, neginf=0)
    target = np.nan_to_num(target, nan=0, posinf=0, neginf=0)
        
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


def rank_features_on_dataset(ds, use_rf=True):
    feature_names = ds.get_feature_names()  # 注意：需要你已修好 idx 前缀化后，这个才严格对齐
    X = ds.raw_train_x[-len(ds.train_y):]
    y = ds.train_y[:, 0].astype(float)

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)

    # -------- 1) pearson (加进度条) --------
    pearson_scores = []
    for i in tqdm(range(X.shape[1]), desc="Pearson", ncols=100):
        try:
            corr, _ = pearsonr(X[:, i], y)
            pearson_scores.append(abs(corr) if np.isfinite(corr) else 0.0)
        except Exception:
            pearson_scores.append(0.0)

    # -------- 2) MI --------
    # MI 没有逐列循环，不太好显示细粒度进度；这里给一个阶段提示即可
    print("[MI] computing mutual information ...")
    mi_scores = mutual_info_regression(X, y)

    # -------- 3) RF importance --------
    if use_rf:
        print("[RF] training random forest for feature importance ... (this can be slow)")
        selected_rf, rf_scores = select_features_by_tree_importance(
            X, y, feature_names, importance_threshold=0.0, print_detail=False
        )
    else:
        rf_scores = np.zeros(X.shape[1], dtype=float)

    df = pd.DataFrame({
        "feature": feature_names[:X.shape[1]],
        "pearson": pearson_scores[:X.shape[1]],
        "mi": mi_scores[:X.shape[1]],
        "rf": rf_scores[:X.shape[1]],
    })

    # 归一化后平均（避免量纲差异）
    for col in ["pearson", "mi", "rf"]:
        mx = float(df[col].max())
        df[col] = df[col] / mx if mx > 1e-12 else df[col]

    df["score"] = df[["pearson", "mi", "rf"]].mean(axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df