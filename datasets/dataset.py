#-*- coding:UTF-8 -*-
import sys,os,time,logging,joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stockinfo import StockInfo
from trade import Trade
from cat import RateCat
from bins import BinManager
from utils.tk import TOKEN
from utils.utils import setup_logging
from utils.utils import StockType
from utils.const_def import CONTINUOUS_DAYS, NUM_CLASSES, T1L_SCALE, T2H_SCALE, BANK_CODE_LIST
from utils.const_def import BASE_DIR, SCALER_DIR, BIN_DIR

# | 数据来源/阶段                 | 变量名                             | 说明                                                       | 数据格式            | 特点/备注                       |
# |-----------------------------|-----------------------------------|-----------------------------------------------------------|--------------------|----------------------------------|
# | T2开始的数据                  | self.full_raw_data                | T2开始的原始数据, 第一列为日期,数据(x)                         | numpy.array        | 不含y1, y2                      |
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
    def __init__(self, ts_code, idx_code_list, rel_code_list, si, start_date=None, end_date=None, window_size=CONTINUOUS_DAYS, train_size=0.8, if_update_scaler=False):
        #logging.debug(f"StockDataset.init - start_date:{start_date}, end_date:{end_date}")
        self.p_trade = Trade(ts_code, si, start_date=start_date, end_date=end_date)
        self.idx_trade_list = [Trade(idx_code, si, stock_type=StockType.INDEX, start_date=start_date, end_date=end_date) for idx_code in idx_code_list]
        self.rel_trade_list = [Trade(rel_code, si, stock_type=StockType.RELATED, start_date=start_date, end_date=end_date) for rel_code in rel_code_list]
        self.if_has_index, self.if_has_related = len(self.idx_trade_list) > 0, len(self.rel_trade_list) > 0
        self.stock = self.p_trade.stock
        self.window_size = window_size
        self.train_size = train_size
        self.if_update_scaler = if_update_scaler
        self.scaler, self.bins1, self.bins2 = None, None, None
        #logging.info(f"if_has_index:{self.if_has_index}, if_has_related:{self.if_has_related}")
        
        #dataset所需的所有数据,都由此方法获取,并存入下列变量
        self.raw_dataset, self.full_raw_data = self.get_trade_data(self.p_trade)  #raw_dataset包含y不含t1t2数据, full_raw_data包含t1t2数据不含y
        self.date_list = self.raw_dataset[:,0]
        #处理关联股票数据
        self.rel_raw_dataset_list, self.rel_full_raw_data_list = zip(*[self.get_trade_data(rel_trade) for rel_trade in self.rel_trade_list]) if self.if_has_related else ([], [])
        logging.info(f"self.raw_dataset shape:{self.raw_dataset.shape}, self.full_raw_data shape:{self.full_raw_data.shape}")
        self.raw_dataset = np.vstack(([self.raw_dataset] + list(self.rel_raw_dataset_list)) if self.if_has_related else self.raw_dataset)
        self.full_raw_data = np.vstack(([self.full_raw_data] + list(self.rel_full_raw_data_list)) if self.if_has_related else self.full_raw_data)
        logging.info(f"self.raw_dataset shape:{self.raw_dataset.shape}, self.full_raw_data shape:{self.full_raw_data.shape}")

        #处理指数数据
        self.idx_raw_dataset_list, self.idx_full_raw_data_list = zip(*[self.get_trade_data(idx_trade) for idx_trade in self.idx_trade_list]) if self.if_has_index else ([], [])
        self.full_raw_data = self.left_merge_np_list(self.full_raw_data, self.idx_full_raw_data_list, col=0) if self.if_has_index else self.full_raw_data
        #过滤指数数据, 只保留与主股票数据日期匹配的行
        self.idx_raw_dataset_list = [self.filter_by_col(self.raw_dataset, idx_raw_data, 0) for idx_raw_data in self.idx_raw_dataset_list] if self.if_has_index else []

        #开始对获取的数据进行加工处理, 形成训练及预测用数据集
        # 1. 分离数据集的x和y
        #   1.1 主股票数据,分离x和y(注意此处还未按窗口处理)
        self.raw_dataset_x, self.raw_y = self.get_dataset_xy(self.raw_dataset)  
        #   1.2 指数数据,分离x和y(注意此处还未按窗口处理)
        self.raw_dataset_x_list, _ = zip(*[self.get_dataset_xy(idx_raw_data) for idx_raw_data in self.idx_raw_dataset_list]) if self.if_has_index else ([], [])
        #   1.3 合并主股票数据和指数数据的x
        self.raw_dataset_x = np.hstack(([self.raw_dataset_x] + list(self.raw_dataset_x_list))) if self.if_has_index else self.raw_dataset_x 

        # 2. 对所有的y一起进行加工处理, 若是多分类模型, 则进行分箱处理, 注意此处未进行窗口化
        self.dataset_y = self.get_binned_y(self.raw_y)  

        # 3. 分离train及test数据
        (self.raw_train_x, self.train_y), (self.raw_test_x, self.test_y) = self.split_train_test_dataset(self.train_size)

        # 4. 根据train_x的数据,生成并保存\读取归一化参数, #根据输入参数判断是否需要更新归一化参数配置,如果更新的话,就保存新的参数配置
        self.scaler = self.get_scaler(new_data=self.raw_train_x, if_update=self.if_update_scaler, if_save=True)  

        # 5. 归一化处理train_x/test_x, 并对x窗口化
        self.normalized_windowed_train_x = self.get_normalized_windowed_x(self.raw_train_x)
        self.normalized_windowed_test_x = self.get_normalized_windowed_x(self.raw_test_x) if train_size < 1 else None

        # 6. 对齐y数据, 因为x按窗口化后会减少数据,所以y也要按窗口大小相应减少
        self.train_y_no_window, self.test_y_no_window = self.train_y, self.test_y #保存未窗口化的y数据,供有需要的使用
        self.train_y, self.test_y = self.train_y[:-self.window_size+1], self.test_y[:-self.window_size+1] #由于x按窗口化后会减少数据,所以y也要相应减少


        logging.info(f"train x/y shape - <{self.normalized_windowed_train_x.shape}/{self.train_y.shape}>")
        logging.info(f"test  x/y shape - <{self.normalized_windowed_test_x.shape}/{self.test_y.shape}>")

    #所有从trade获取的数据都由此方法返回
    def get_trade_data(self, trade):
        return trade.combine_data_np, trade.raw_data_np

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
    #直接返回原始y数据(涨跌幅),不进行分箱
    def get_binned_y(self, raw_y):
        return raw_y*100
    
    #按本dataset的y数据,生成对应的分箱器,并返回分箱后的y数据
    #此处提供两种分箱方法,可选其一
    def get_binned_y_use_qcut(self, raw_y):
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        self.bins1 = BinManager(y1, n_bins=NUM_CLASSES, save_path=os.path.join(BASE_DIR, BIN_DIR, self.stock.ts_code + "_y1_bins.json"))
        self.bins2 = BinManager(y2, n_bins=NUM_CLASSES, save_path=os.path.join(BASE_DIR, BIN_DIR, self.stock.ts_code + "_y2_bins.json"))
        y1_binned = np.array([RateCat(rate=x,scale=self.bins1.prop_bins,right=True).get_label() for x in y1])
        y2_binned = np.array([RateCat(rate=x,scale=self.bins2.prop_bins,right=True).get_label() for x in y2])
        return (np.array([y1_binned, y2_binned]).astype(int)).transpose()
    #按手工生成具体的分箱,并返回分箱后的y数据
    #此处提供两种分箱方法,可选其一
    def get_binned_y_use_scale(self, raw_y):
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        self.bins1 = np.array([-np.inf] + T1L_SCALE + [np.inf])
        self.bins2 = np.array([-np.inf] + T2H_SCALE + [np.inf])
        y1_binned = np.digitize(y1, bins=T1L_SCALE, right=True)
        y2_binned = np.digitize(y2, bins=T2H_SCALE, right=True)
        return (np.array([y1_binned, y2_binned]).astype(int)).transpose()

    #划分数据集(注意此处数据还未按窗口处理)
    #返回格式:
    # (train_x, train_y), (test_x, test_y) 
    def split_train_test_dataset(self, train_size):
        test_size = 1-train_size
        test_count = int(len(self.raw_dataset_x) * test_size)
        raw_test_x, raw_train_x = np.array(self.raw_dataset_x[:test_count]).astype(float), np.array(self.raw_dataset_x[test_count:]).astype(float)
        test_y, train_y = self.dataset_y[:test_count,:], self.dataset_y[test_count:,:]
        logging.debug(f"{len(self.raw_dataset_x)} rows of data, train_size:{train_size:.2f}, test_size:{test_size:.2f}, test_count:{test_count:d}")
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
        closed_price = self.full_raw_data[idx, self.p_trade.col_close+1] #取出对应日期的收盘价, +1是因为full_raw_data含日期列
        #logging.info(f"full_raw_data \n{pd.DataFrame(self.full_raw_data[idx])}")
        raw_x = self.full_raw_data[idx:idx+self.window_size, 1:]#取出对应日期及之后window_size天的数据
        x = self.get_normalized_windowed_x(raw_x) #归一化,窗口化
        return x, closed_price
    
    #按指定列进行左连接
    # on - 指定连接的列,如['trade_date']
    def left_merge_np(self, left, right, col):
        df_left, df_right = pd.DataFrame(left), pd.DataFrame(right)
        left_col_name, right_col_name = df_left.columns[col], df_right.columns[col]
        df_merged = pd.merge(df_left, df_right, left_on=left_col_name, right_on=right_col_name, how='left')
        col_to_drop = df_merged.columns[df_left.shape[1]]  #取出左表中用于连接的列名
        df_merged = df_merged.drop(columns=[col_to_drop])  if left_col_name != right_col_name else df_merged
        return df_merged.to_numpy()

    #按指定列对一批数据进行左连接
    # on - 指定连接的列,如['trade_date']
    def left_merge_np_list(self, left, data_list, col):
        #pd.set_option('display.max_columns', None)
        if len(data_list) < 1:
            logging.error("Invalid parameters.")
            exit()
        left_df = pd.DataFrame(left)
        for right in data_list:
            left = self.left_merge_np(left, right, col)
        return left

    def filter_by_col(self, a: np.ndarray, b: np.ndarray, col_no: int) -> np.ndarray:
        """
        以 a 的指定列(col_no)为基准, 过滤 b, 只保留 b 指定列与 a 指定列匹配的行。
        参数:
            a: numpy二维数组, 作为基准
            b: numpy二维数组, 被过滤
        返回:
            筛选后的b (np.ndarray)
        """
        base_col = a[:, col_no].astype(np.int64)  #确保类型一致
        mask = np.isin(b[:, col_no].astype(np.int64), base_col)
        return b[mask]

    def get_feature_names(self):
        """
        返回主股票和指数合并后的特征名列表（不包含日期、ts_code）。
        """
        feature_names = []
        feature_names += list(self.p_trade.trade_df.columns.drop(['ts_code','trade_date'], errors='ignore'))
        for idx_trade in getattr(self, 'idx_trade_list', []):
            feature_names += list(idx_trade.trade_df.columns.drop(['ts_code','trade_date'], errors='ignore'))
        return feature_names

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
    primary_stock_code = '600036.SH'
    idx_code_list = []#'000001.SH','399001.SZ']#,'000300.SH','000905.SH']
    rel_code_list = BANK_CODE_LIST
    ds = StockDataset(primary_stock_code, idx_code_list, rel_code_list, si, start_date='20070104', end_date='20250903', train_size=0.8)
    pd.set_option('display.max_columns', None)
    print(pd.DataFrame(ds.normalized_windowed_train_x[0]).head(5))
    #ds.print_comp_data()
    #logging.info(f"SHAPE - train_y:{ds.train_y.shape}, test_y:{ds.test_y.shape}")
    #logging.info(ds.test_y)
