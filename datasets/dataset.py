#-*- coding:UTF-8 -*-
import sys,os,time,logging,joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stock import Stock
from stockinfo import StockInfo
from trade import Trade
from cat import RateCat
from bins import BinManager
from utils.utils import setup_logging
from utils.const_def import TOKEN, CONTINUOUS_DAYS, NUM_CLASSES
from utils.const_def import BASE_DIR, SCALER_DIR, BIN_DIR

# | 数据来源/阶段                 | 变量名                             | 说明                                                       | 数据格式            | 特点/备注                       |
# |-----------------------------|-----------------------------------|-----------------------------------------------------------|--------------------|----------------------------------|
# | T2开始的数据                  | self.full_raw_data                | T2开始的原始数据, 第一列为ts_code, 第二列为日期                 | numpy.array        | 不含y1, y2                      |
# | 不含T2的数据                  |                                   |                                                           |                    |                                  |
# | 未分箱, 未窗口化的数据          | self.date_list                    | 只有数据集日期的一维数组                                      | numpy.array        |                                  |
# |                             | self.raw_dataset                  | 包含日期(第一列), 数据(x), 训练预测值(y)的二维数组               | numpy.array        |                                  |
# |                             | self.raw_dataset_x                | 不含日期, 数据集的x和y(注意此处还未按窗口处理)                   | numpy.array        |                                  |
# |                             | self.raw_y                        | y1, y2的原始值, 列数为2的二维数组                             | numpy.array        |                                  |
# |                             | self.raw_train_x                  | 划分后的训练和测试集(注意此处数据还未按窗口处理), y已分箱          | numpy.array        |                                  |
# |                             | self.raw_test_x                   | 划分后的训练和测试集(注意此处数据还未按窗口处理), y已分箱          | numpy.array        |                                  |
# | 分箱后的y                    |  self.train_y, self.test_y         | 分箱后的y, 列数为2的二维数组                                  | numpy.array        |                                  |
# | 归一化, 窗口化后的x            | self.normalized_windowed_train_x  | 归一化, 窗口化后的训练集x, 三维数组                             | numpy.array        |                                  |
# |                             | self.normalized_windowed_test_x   | 归一化, 窗口化后的测试集x, 三维数组                             | numpy.array        |                                  |
class StockDataset():
    def __init__(self, ts_code, si, start_date=None, end_date=None, window_size=CONTINUOUS_DAYS, train_size=0.8, if_update_scaler=False):
        logging.debug(f"StockDataset.init - start_date:{start_date}, end_date:{end_date}")
        self.trade = Trade(ts_code, si, start_date=start_date, end_date=end_date)
        self.stock = self.trade.stock
        self.window_size = window_size
        self.train_size = train_size
        self.if_update_scaler = if_update_scaler
        self.scaler, self.bins1, self.bins2 = None, None, None
        
        #dataset所需的所有数据,都由此方法获取,并存入下列3个变量
        self.raw_dataset, self.full_raw_data = self.get_trade_data(self.trade.trade_df)  #raw_dataset包含日期, full_raw_data不含日期
        self.date_list = self.raw_dataset[:,0]

        #开始对获取的数据进行加工处理,形成训练及预测用数据集
        # 1. 分离数据集的x和y
        self.raw_dataset_x, self.raw_y = self.get_dataset_xy(self.raw_dataset)  #数据集的x和y(注意此处还未按窗口处理)

        # 2. 对所有的y一起进行分箱处理
        self.dataset_y = self.get_binned_y(self.raw_y)  #y已分箱,但还未按窗口处理

        # 3. 分离train及test数据
        (self.raw_train_x, self.train_y), (self.raw_test_x, self.test_y) = self.split_train_test_dataset()

        # 4. 根据train_x的数据,生成并保存\读取归一化参数, #根据输入参数判断是否需要更新归一化参数配置,如果更新的话,就保存新的参数配置
        self.scaler = self.get_scaler(new_data=self.raw_train_x, if_update=self.if_update_scaler, if_save=True)  

        # 5. 归一化处理train_x/test_x, 并对x窗口化
        self.normalized_windowed_train_x = self.get_normalized_windowed_x(self.raw_train_x)
        self.normalized_windowed_test_x = self.get_normalized_windowed_x(self.raw_test_x) if train_size < 1 else None

        # 6. 对齐y数据, 因为x按窗口化后会减少数据,所以y也要按窗口大小相应减少
        self.train_y, self.test_y = self.train_y[:-self.window_size+1], self.test_y[:-self.window_size+1] #由于x按窗口化后会减少数据,所以y也要相应减少

        logging.info(f"train x/y shape - <{self.normalized_windowed_train_x.shape}/{self.train_y.shape}>")
        logging.info(f"test  x/y shape - <{self.normalized_windowed_test_x.shape}/{self.test_y.shape}>")

    #所有从trade获取的数据都由此方法返回
    def get_trade_data(self, trade_df):
        return trade_df.combine_data_np, trade_df.raw_data_np

    #返回数据集的x和y(注意此处还未按窗口处理)
    #格式:
    #dataset_x --
    # [ [x_a_1,x_b_1,...],
    #   [x_a_2,x_b_2,...],
    #   ... ]
    #dataset_y --
    # [ [y_a_1,y_b_1],
    #   [y_a_2,y_b_2],
    #   ... ]
    def get_dataset_xy(self, raw_data):
        dataset_all = np.delete(raw_data, 0, axis=1) #删除日期列
        dataset_x = dataset_all[:, :-2] #删除最后2列
        dataset_y = dataset_all[:, -2:] #取最后2列
        return np.array(dataset_x).astype(float), np.array(dataset_y).astype(float)

    #按本dataset的y数据,生成对应的分箱器,并返回分箱后的y数据
    def get_binned_y(self, raw_y):
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        self.bins1 = BinManager(y1, n_bins=NUM_CLASSES, save_path=os.path.join(BASE_DIR, BIN_DIR, self.stock.ts_code + "_y1_bins.json"))
        self.bins2 = BinManager(y2, n_bins=NUM_CLASSES, save_path=os.path.join(BASE_DIR, BIN_DIR, self.stock.ts_code + "_y2_bins.json"))
        y1_binned = np.array([RateCat(rate=x,scale=self.bins1.prop_bins,right=True).get_label() for x in y1])
        y2_binned = np.array([RateCat(rate=x,scale=self.bins2.prop_bins,right=True).get_label() for x in y2])
        return (np.array([y1_binned, y2_binned]).astype(int)).transpose()

    #划分数据集(注意此处数据还未按窗口处理)
    #返回格式:
    # (train_x, train_y), (test_x, test_y) 
    def split_train_test_dataset(self):
        test_size = 1-self.train_size
        test_count = int(len(self.raw_dataset_x) * test_size)
        raw_test_x, raw_train_x = np.array(self.raw_dataset_x[:test_count]).astype(float), np.array(self.raw_dataset_x[test_count:]).astype(float)
        test_y, train_y = self.dataset_y[:test_count,:], self.dataset_y[test_count:,:]
        logging.debug(f"{len(self.raw_dataset_x)} rows of data, train_size:{self.train_size:.2f}, test_size:{test_size:.2f}, test_count:{test_count:d}")
        return (raw_train_x, train_y), (raw_test_x, test_y)

    #归一化处理数据
    def get_normalized_data(self, raw_data):
        if raw_data is not None:
            df = pd.DataFrame(raw_data)
            scaled_data = self.scaler.transform(df)
        else:
            logging.error("Invalid parameters.")
            exit()
        return np.array(scaled_data).astype(float)
    
    #获取归一化参数
    def get_scaler(self, new_data=None, if_update=False, if_save=True):
        is_modified = False
        self.scaler = RobustScaler()#StandardScaler()

        if new_data is not None or not os.path.exists(os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + "_scaler.save")):    #如果有新的数据,则更新归一化的参数配置
            df = pd.DataFrame(new_data)
            self.scaler.fit(df)
            is_modified = True
            if if_save and is_modified: #如果有更新且需要保存,则保存
                joblib.dump(self.scaler, os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + '_scaler.save'))
                logging.info(f"write scaler cfg to {os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + '_scaler.save')}")
        else:   #如果没有输入数据参数,则直接读取已保存的归一化参数配置
            try:
                self.scaler = joblib.load(os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + "_scaler.save"))
                logging.info(f"load scaler from {os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + '_scaler.save')}")
            except Exception as e:
                logging.error(f"StockDataset.get_real_normalize() - Failed to load scaler: {e}")
                exit()
        return self.scaler

    #将输入的二维x,按窗口大小处理成三维的数据
    # 输出x格式 - [样本, 时间步, 特征]
    def get_windowed_x_by_raw(self, dataset_x=None):
        x = []
        for i in range(len(dataset_x) - self.window_size + 1):
            x_window=[]
            for ii in range(self.window_size):
                x_window.append(dataset_x[i + ii])
            x.append(x_window)
        return np.array(x).astype(float)
    
    #获取归一化,窗口化后的x数据
    def get_normalized_windowed_x(self, raw_x):
        self.get_scaler() if self.scaler is None else None
        normalized_x = self.get_normalized_data(raw_x)
        return self.get_windowed_x_by_raw(normalized_x)

    #获取某一天的模型输入数据(已归一化,无法获取T1,T2的数据),以及对应的真实结果y
    def get_dataset_with_y_by_date(self, date):
        date = type(self.full_raw_data[0, 0])(date)
        try:
            idx = np.where(self.date_list == date)[0][0]
        except Exception as e:
            logging.error(f"StockDataset.get_dataset_by_date() - Invalid date: {e}")
            exit()
        return self.normalized_windowed_train_x[idx,:,:] , self.raw_y[idx]
    
    #获取某一天的模型输入数据(已归一化,可以获取T2的数据)
    def get_predictable_dataset_by_date(self, date):
        date = type(self.full_raw_data[0, 0])(date)
        try:
            idx = np.where(self.full_raw_data[:, 0] == date)[0][0]
        except Exception as e:
            logging.error(f"StockDataset.get_predictable_dataset_by_date() - Invalid date: {e}")
            exit()
        closed_price = self.full_raw_data[idx, 4]
        raw_x = self.full_raw_data[idx:idx+self.window_size, 1:]#取出对应日期及之后window_size天的数据
        x = self.get_normalized_windowed_x(raw_x) #归一化,窗口化
        return x, closed_price

    #打印归一化前后的数据对比
    def print_comp_data(self, i=0, col=47):
        tx = self.normalized_windowed_train_x
        raw_x = self.get_windowed_x_by_raw(self.raw_dataset_x)
        logging.info(f"raw_x shape:{raw_x.shape}, raw_x:\n{raw_x[0,0]}")
        logging.info(f"print_comp_data - i:[{i}], col:[{col}]")
        for idx, (val, val_raw) in enumerate(zip(tx[i,:,col], raw_x[i,:,col])):
            logging.info(f"[{self.date_list[idx+i]}] val/raw_val <{val:+.5f}/{val_raw}>")


if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    #download_list = si.get_filtered_stock_list(mmv=3000000)
    primary_stock = '600036.SH'
    ds = StockDataset(primary_stock, si, start_date='20250101', end_date='20250903', train_size=0.8)
    #ds.print_comp_data()
    #logging.info(f"SHAPE - train_y:{ds.train_y.shape}, test_y:{ds.test_y.shape}")
    #logging.info(ds.test_y)
