#-*- coding:UTF-8 -*-
import sys,os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from deprecated import deprecated
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from dataset import StockDataset
from cat import RateCat
from bins import BinManager
from utils.const_def import BASE, NUM_CLASSES

class StockDatasets():
    def __init__(self, primary_stock, related_stocks, train_size=0.8, if_update_scaler=False):
        self.primary_stock = primary_stock
        self.related_stocks = related_stocks
        self.primary_stockdataset = StockDataset(primary_stock, train_size=train_size, if_update_scaler=True)
        self.related_stockdatasets = []
        for stock in self.related_stocks.stock_list:
            self.related_stockdatasets.append(StockDataset(stock,train_size=1, if_update_scaler=if_update_scaler))
        self.date_list = self.related_stockdatasets[0].trade.trade_date_list if self.related_stockdatasets else []
        
        self.bins1, self.bins2 = self.get_bins()
        self.train_x = self.get_normalized_windowed_x()
        self.test_x = self.primary_stockdataset.normalized_windowed_test_x
        self.train_y, self.test_y = self.get_y(train_size=train_size)   ##注意此处的所有y数据,都要根据全数据集重新分箱生成
        #self.train_y1, self.train_y2 = self.train_y[:,0], self.train_y[:,1]
        #self.test_y1, self.test_y2 = self.test_y[:,0], self.test_y[:,1]
        print("INFO: bins1 - %s" % str(self.bins1.prop_bins))
        print("INFO: train x/y shape - <%s/%s>"%(str(self.train_x.shape),str(self.train_y.shape)))
        print("INFO: test  x/y shape - <%s/%s>"%(str(self.test_x.shape),str(self.test_y.shape)))
    
    def get_normalized_windowed_x(self):
        x_full = self.primary_stockdataset.normalized_windowed_train_x
        for dataset in self.related_stockdatasets:
            x_full = np.concatenate((x_full, dataset.normalized_windowed_train_x), axis=0)
        return x_full
    
    def get_raw_y(self):
        y_full = self.primary_stockdataset.raw_y
        for dataset in self.related_stockdatasets:
            y_full = np.concatenate((y_full, dataset.raw_y), axis=0)
        return y_full
    
    def get_bins(self):
        raw_y = self.get_raw_y()
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        self.bins1 = BinManager(y1, n_bins=NUM_CLASSES, save_path=BASE + "\\cfg\\" + self.primary_stock.ts_code + "_pri_y1_bins.json")
        self.bins2 = BinManager(y2, n_bins=NUM_CLASSES, save_path=BASE + "\\cfg\\" + self.primary_stock.ts_code + "_pri_y2_bins.json")
        return self.bins1, self.bins2

    def get_binned_y(self, bins1, bins2, y):
        ret_y1 = np.array([RateCat(rate=x,scale=bins1.prop_bins, right=True).get_label() for x in y[:, 0]])
        ret_y2 = np.array([RateCat(rate=x,scale=bins2.prop_bins, right=True).get_label() for x in y[:, 1]])
        return (np.array([ret_y1, ret_y2]).astype(int)).transpose()

    def get_y(self, train_size):
        
        primary_stock_test_cnt = int(len(self.primary_stockdataset.raw_dataset) * (1-train_size))
        y_test = self.get_binned_y(self.bins1, self.bins2, self.primary_stockdataset.raw_y[:primary_stock_test_cnt-self.primary_stockdataset.window_size+1])    #减少对应的窗口数量
        y_train = self.get_binned_y(self.bins1, self.bins2, self.primary_stockdataset.raw_y[primary_stock_test_cnt:-self.primary_stockdataset.window_size+1])     #减少对应的窗口数量
        for dataset in self.related_stockdatasets:
            y = self.get_binned_y(self.bins1, self.bins2, dataset.raw_y[:-self.primary_stockdataset.window_size+1])
            y_train = np.concatenate((y_train, y), axis=0)
        return y_train, y_test

    def get_raw(self):
        x_full = self.primary_stockdataset.raw_dataset
        for dataset in self.related_stockdatasets:
            x_full = np.concatenate((x_full, dataset.raw_dataset), axis=0)
        return x_full

    def save_raw(self, file_path=None):
        if file_path is None or file_path == "":
            fn = BASE+'\\temp\\raw_x.csv'
        else:
            fn = file_path
        arr = self.get_raw()
        df = pd.DataFrame(arr)
        df.to_csv(fn, index=False)

    def label_ratio(self):
        """
        统计标签类别占比
        参数:
            y: 标签数组或Series, 例如 [0, 1, 2, 0, 1, 2, 2]
        返回:
            dict, 每个类别及其占比
        """
        # 如果是numpy数组或列表，转换为pandas Series
        y1 = pd.Series(self.train_y[:,0])
        y2 = pd.Series(self.train_y[:,1])
        # 统计各类别数量
        y1_counts = y1.value_counts()
        y2_counts = y2.value_counts()
        # 计算总数
        total1 = len(y1)
        total2 = len(y2)
        # 计算占比
        ratio1 = (y1_counts / total1).to_dict()
        ratio2 = (y2_counts / total2).to_dict()
        return ratio1, ratio2
