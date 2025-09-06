# coding=utf-8
import os, sys, time, argparse, datetime, logging
import pandas as pd
import numpy as np
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from utils.tk import TOKEN
from utils.const_def import REL_CODE_LIST, NUM_CLASSES
from utils.utils import setup_logging, StockType
from datasets.bins import BinManager
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bin_feature_correlation(bin_labels, feature_data, feature_names=None, show=True, save_path=None):
    """
    绘制分箱与特征均值的相关性热力图或柱状图
    参数:
        bin_labels: 一维分箱编号数组（如 [0,1,1,2,...]），长度为样本数
        feature_data: 二维特征数组(shape: [样本数, 特征数])
        feature_names: 特征名列表(如 ['close', 'volume', ...])
        show: 是否显示图像
        save_path: 保存路径
    """

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(feature_data.shape[1])]
    df = pd.DataFrame(feature_data, columns=feature_names)
    df['bin'] = bin_labels

    # 统计每个分箱的各特征均值
    mean_df = df.groupby('bin').mean().T

    # 热力图
    plt.figure(figsize=(max(10, feature_data.shape[1]//2), 6))
    sns.heatmap(mean_df, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Feature Mean by Bin')
    plt.xlabel('Bin')
    plt.ylabel('Feature')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

def get_suggested_features(corr_matrix, feature_names, threshold=0.05):
    # 计算每个特征在各分箱均值的极差（最大-最小），作为区分度指标
    ranges = np.ptp(corr_matrix, axis=1)
    keep_idx = np.where(ranges > threshold)[0]
    drop_idx = np.where(ranges <= threshold)[0]
    keep_names = [feature_names[i] for i in keep_idx]
    drop_names = [feature_names[i] for i in drop_idx]
    return keep_names, drop_names
    
if __name__ == "__main__":
    setup_logging()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    epo_list = [200]
    #epo_list = [10,50,100]
    p_list = [4]
    #p_list = [2,16]
    batch_size_list = [32]
    #batch_size_list = [8,128,512,32]
    if_print_detail = False

    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = ['000001.SH', '399001.SZ', '399006.SZ']  #上证指数,深证成指,创业板指
    related_stock_list = REL_CODE_LIST
    ds = StockDataset(primary_stock_code, index_code_list, si, start_date='20100601',end_date='20250903', train_size=0.9)
    
    feature_names = ds.get_feature_names()

    # x: (样本数, 时间步, 特征)
    windowed_x = ds.normalized_windowed_train_x.mean(axis=1)  # shape: (样本数, 特征数)
    bins1 = ds.train_y[:,0]  # shape: (样本数,)
    bin_manager = BinManager()
    bin_manager.plot_bin_feature_correlation(bins1, windowed_x, feature_names=feature_names, show=True, save_path="bin_feature_corr.png")

    df = pd.DataFrame(windowed_x, columns=feature_names)
    df['bin'] = bins1
    mean_df = df.groupby('bin').mean().T  # [特征, 分箱]
    corr_matrix = mean_df.values  # 取出为numpy array

    print(corr_matrix.shape)  # (特征数, 分箱数)
    
    # 用法示例
    # corr_matrix = ... # shape: [61, 20]
    # feature_names = [...] # 61个特征名
    keep, drop = get_suggested_features(corr_matrix, feature_names)

    print("建议保留：", keep)
    print("建议删除：", drop)