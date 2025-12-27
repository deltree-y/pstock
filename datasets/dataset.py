#-*- coding:UTF-8 -*-
import sys,os,time,logging,joblib
import numpy as np
import pandas as pd
#from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
#o_path = os.getcwd()
#sys.path.append(o_path)
#sys.path.append(str(Path(__file__).resolve().parents[0]))
from stockinfo import StockInfo
from trade import Trade
from cat import RateCat
from bins import BinManager
from utils.tk import TOKEN
from utils.utils import FeatureType, setup_logging
from utils.utils import StockType, PredictType
from utils.const_def import CONTINUOUS_DAYS, NUM_CLASSES, MIN_TRADE_DATA_ROWS, T1L_SCALE, T2H_SCALE, ALL_CODE_LIST, IDX_CODE_LIST, CLIP_Y_PERCENT
from utils.const_def import BASE_DIR, SCALER_DIR, BIN_DIR

# | 数据来源/阶段                 | 变量名                             | 说明                                                       | 数据格式            | 特点/备注                       |
# |-----------------------------|-----------------------------------|-----------------------------------------------------------|--------------------|----------------------------------|
# | T2开始的数据                  | self.raw_data                     | T2开始的原始数据, 第一列为日期,数据(x)                         | numpy.array        | 不含y1, y2                      |
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
    def __init__(self, ts_code, idx_code_list, rel_code_list, si, start_date=None, end_date=None, train_size=0.8, if_update_scaler=True, 
                 feature_type:FeatureType=FeatureType.ALL, predict_type:PredictType=PredictType.CLASSIFY, use_conv2_channel=False):
        self.p_trade = Trade(ts_code, si, start_date=start_date, end_date=end_date, feature_type=feature_type)
        self.idx_trade_list = [Trade(idx_code, si, stock_type=StockType.INDEX, start_date=start_date, end_date=end_date, feature_type=feature_type) for idx_code in idx_code_list]
        self.rel_trade_list = []
        for rel_code in rel_code_list:
            t = Trade(rel_code, si, stock_type=StockType.RELATED, start_date=start_date, end_date=end_date, feature_type=feature_type)
            self.rel_trade_list.append(t) if t.trade_count >= MIN_TRADE_DATA_ROWS else None #剔除数据过少的关联股票
        self.if_has_index, self.if_has_related = len(self.idx_trade_list) > 0, len(self.rel_trade_list) > 0
        self.si = si
        self.stock = self.p_trade.stock
        self.window_size = CONTINUOUS_DAYS
        self.train_size = train_size
        self.if_update_scaler = if_update_scaler
        self.feature_type = feature_type
        self.predict_type = predict_type
        self.use_conv2_channel = use_conv2_channel  # 新增参数

        self.scaler, self.bins1, self.bins2 = None, None, None
        self.y_cnt = self.p_trade.y_cnt

        if False:
            # ================= 支持 channel 模式 BEGIN =================
            if self.use_conv2_channel:
                all_trades = [self.p_trade] + self.rel_trade_list
                idx_raw_data_list = [t.raw_data_np for t in self.idx_trade_list] if self.if_has_index else []
                raw_dataset_list = []   # 每个为[日期, features..., y1, y2, ...]
                for trade in all_trades:
                    dataset, _ = self.get_trade_data(trade)
                    for idx_raw_data in idx_raw_data_list:
                        dataset = self.left_join_pd_with_move_last(dataset, idx_raw_data, move_last_n_cols=self.y_cnt)
                    raw_dataset_list.append(dataset)
                # date_list、raw_data都可以用主股票的
                self.date_list = raw_dataset_list[0][:, 0]
                self.raw_data = raw_dataset_list[0]
                
                # 自动适配任务分割
                (raw_train_x, train_y, raw_train_y), (raw_test_x, test_y, raw_test_y) = \
                    self.split_train_test_dataset_by_stock(raw_dataset_list, self.train_size)
                self.raw_train_x, self.train_y, self.raw_train_y = raw_train_x, train_y, raw_train_y
                self.raw_test_x, self.test_y, self.raw_test_y = raw_test_x, test_y, raw_test_y

                # 分箱或任务相关属性
                self.raw_y = self.get_y_from_raw_dataset(np.vstack(raw_dataset_list))
                self.bins1, self.bins2 = self.get_bins(self.raw_y)

                # fit scaler
                #feature_x_list = [x[:, 1:-self.y_cnt] if self.y_cnt > 0 else x[:, 1:] for x in self.raw_train_x]
                concat_train_x = np.vstack(self.raw_train_x)
                self.scaler = self.get_scaler(new_data=concat_train_x, if_update=self.if_update_scaler, if_save=True)

                # 归一化+窗口化（shape: [样本, 时间步, feature, channel]）
                #feature_x_list = [x[:, 1:-self.y_cnt] if self.y_cnt > 0 else x[:, 1:] for x in self.raw_train_x]
                self.normalized_windowed_train_x = self.get_normalized_windowed_x(self.raw_train_x)
                #feature_x_test_list = [x[:, 1:-self.y_cnt] if self.y_cnt > 0 else x[:, 1:] for x in self.raw_test_x]
                self.normalized_windowed_test_x = self.get_normalized_windowed_x(self.raw_test_x) if train_size < 1 else None

                # y 对齐窗口
                self.train_y = self.train_y[:-self.window_size + 1]
                if self.test_y is not None and self.normalized_windowed_test_x is not None:
                    self.test_y = self.test_y[:-self.window_size + 1]

                return
            # ================= 支持 channel 模式 END ===================

        #处理主股票数据
        self.raw_dataset, self.raw_data = self.get_trade_data(self.p_trade)  #raw_dataset包含y不含t1t2数据, raw_data包含t1t2数据不含y
        self.date_list = self.raw_dataset[:,0]
        self.raw_date_list = self.raw_data[:,0]
        #处理关联股票数据
        self.rel_raw_dataset_list, self.rel_raw_data_list = zip(*[self.get_trade_data(rel_trade) for rel_trade in self.rel_trade_list]) if self.if_has_related else ([], [])
        # ===== Debug: 打印各通道长度 & 最短通道 ts_code =====
        channel_trades = [self.p_trade] + list(self.rel_trade_list)
        channel_lens = [t.combine_data_np.shape[0] for t in channel_trades]
        min_len = min(channel_lens) if channel_lens else 0
        min_idx = channel_lens.index(min_len) if channel_lens else 0
        logging.info(f"[channel length] { {t.ts_code: l for t, l in zip(channel_trades, channel_lens)} }"
                     f" -> shortest: {channel_trades[min_idx].ts_code if channel_trades else 'N/A'} (len={min_len})")
        # ====== ================ Debug END ==================

        self.raw_data = np.vstack(([self.raw_data] + list(self.rel_raw_data_list)) if self.if_has_related else self.raw_data)   #已包括主股票及关联股票数据
        #处理指数数据
        self.idx_raw_dataset_list, self.idx_raw_data_list = zip(*[self.get_trade_data(idx_trade) for idx_trade in self.idx_trade_list]) if self.if_has_index else ([], [])
        if self.if_has_index:
            for idx_raw_data in self.idx_raw_data_list: #逐个将指数数据并接到主股票及关联股票数据上, 如果批量处理的话, 会打乱已有的raw_data顺序
                self.raw_data = self.left_join_pd_with_move_last(self.raw_data, idx_raw_data, if_debug=False) if self.if_has_index else self.raw_data    #已包括主股票及关联股票及指数数据

        ###########      ********************      #################
        ### 每只股票单独切分训练/测试，再合并 ###
        # 1. 合并主股票与相关联股票的原始数据
        raw_dataset_list = [self.raw_dataset] + list(self.rel_raw_dataset_list) if self.if_has_related else [self.raw_dataset]

        # 1.5 基于所有的原始y数据生成分箱器
        self.raw_y = self.get_y_from_raw_dataset(np.vstack(raw_dataset_list)) #取出y(多个)
        self.bins1, self.bins2 = self.get_bins(self.raw_y)

        # 2. 按股票分离测试集与验证集,并对y进行分箱,返回对应的y
        #(self.raw_train_x, self.train_y, self.raw_train_y), (self.raw_test_x, self.test_y, self.raw_test_y) = \
        #    self.split_train_test_dataset_by_stock(raw_dataset_list, self.train_size)
        (self.raw_train_x, self.train_y, self.raw_train_y), (self.raw_test_x, self.test_y, self.raw_test_y) = self.split_train_test_dataset_by_stock(
            raw_dataset_list, self.train_size, collect_aligned_raw=self.use_conv2_channel
        ) 

        # 4. 根据train_x的数据,生成并保存\读取归一化参数, #根据输入参数判断是否需要更新归一化参数配置,如果更新的话,就保存新的参数配置
        #self.scaler = self.get_scaler(new_data=self.raw_train_x, if_update=self.if_update_scaler, if_save=True)  
        self.scaler = self.get_scaler(
            new_data=self._stack_features_for_scaler(self.raw_train_x),
            if_update=self.if_update_scaler,
            if_save=True
        )

        # 5. 归一化处理train_x/test_x, 并对x窗口化
        self.normalized_windowed_train_x = self.get_normalized_windowed_x(self.raw_train_x)
        self.normalized_windowed_test_x = self.get_normalized_windowed_x(self.raw_test_x) if train_size < 1 else None

        # 6. 对齐y数据, 因为x按窗口化后会减少数据,所以y也要按窗口大小相应减少
        self.train_y_no_window, self.test_y_no_window = self.train_y, self.test_y #保存未窗口化的y数据,供有需要的使用
        self.train_y, self.test_y = self.train_y[:-self.window_size+1], self.test_y[:-self.window_size+1] #由于x按窗口化后会减少数据,所以y也要相应减少

        #logging.info(f"train x/y shape - <{self.normalized_windowed_train_x.shape}/{self.train_y.shape}>")
        #logging.info(f"test  x/y shape - <{self.normalized_windowed_test_x.shape}/{self.test_y.shape}>")

    #所有从trade获取的数据都由此方法返回
    def get_trade_data(self, trade):
        return trade.combine_data_np, trade.raw_data_np
    
    #从原始数据集中取出y数据
    def get_y_from_raw_dataset(self, raw_dataset):
        return raw_dataset[:, -1*self.y_cnt:].astype(float) #取出最后n列作为y

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
        dataset_x = dataset_all[:, :-1*self.y_cnt] #删除最后n列
        dataset_y = dataset_all[:, -1*self.y_cnt:] #取最后n列
        return np.array(dataset_x).astype(float), np.array(dataset_y).astype(float)


    #按本dataset的y数据,生成对应的分箱器,并返回分箱后的y数据
    #直接返回原始y数据(涨跌幅),不进行分箱
    def get_binned_y(self, raw_y):
        return raw_y*100

    #此处提供两种分箱方法,可选其一
    #方法一(自动分箱) : 按本dataset的y数据,生成对应的分箱器,并返回分箱后的y数据
    def get_bins(self, raw_y):
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        bins1 = BinManager(y1, n_bins=NUM_CLASSES, save_path=os.path.join(BASE_DIR, BIN_DIR, self.stock.ts_code + "_y1_bins.json"))
        bins2 = BinManager(y2, n_bins=NUM_CLASSES, save_path=os.path.join(BASE_DIR, BIN_DIR, self.stock.ts_code + "_y2_bins.json"))
        return bins1, bins2
    def get_binned_y_use_qcut(self, raw_y):
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        if self.bins1 is not None and self.bins2 is not None:
            y1_binned = np.array([RateCat(rate=y,scale=self.bins1.bins,right=True).get_label() for y in y1])
            y2_binned = np.array([RateCat(rate=y,scale=self.bins2.bins,right=True).get_label() for y in y2])
            return (np.array([y1_binned, y2_binned]).astype(int)).transpose()
        logging.error("StockDataset.get_binned_y_use_qcut() - bins1 or bins2 is None.")
        exit()
        
    #方法二(手工分箱) : 按手工生成具体的分箱,并返回分箱后的y数据
    def get_binned_y_use_scale(self, raw_y):
        y1,y2 = raw_y[:, 0], raw_y[:, 1]
        self.bins1 = np.array([-np.inf] + T1L_SCALE + [np.inf])
        self.bins2 = np.array([-np.inf] + T2H_SCALE + [np.inf])
        y1_binned = np.digitize(y1, bins=T1L_SCALE, right=True)
        y2_binned = np.digitize(y2, bins=T2H_SCALE, right=True)
        return (np.array([y1_binned, y2_binned]).astype(int)).transpose()

    # 新增：每只股票单独切分训练/测试集，再合并
    def split_train_test_dataset_by_stock(self, raw_dataset_list, train_size, collect_aligned_raw=False):
        train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
        raw_train_y_list, raw_test_y_list = [], [] #保存未分箱的y数据,供有需要的使用
        aligned_raw_list = [] if collect_aligned_raw else None  #for conv2d add
        ch_train_x_list, ch_test_x_list = [], []  # 每通道切分后的X（含日期+y，后续再裁剪特征）
        for raw_data in raw_dataset_list:
            raw_data_with_idx = raw_data 
            if self.if_has_index:
                for idx_raw_data in self.idx_raw_data_list:
                    raw_data_with_idx = self.left_join_pd_with_move_last(raw_data_with_idx, idx_raw_data, move_last_n_cols=self.y_cnt) if self.if_has_index else raw_data    #并接指数数据
            if collect_aligned_raw:                         #for conv2d add
                aligned_raw_list.append(raw_data_with_idx)  #for conv2d add
            raw_x, raw_y = self.get_dataset_xy(raw_data_with_idx)
            if len(raw_x) < NUM_CLASSES:
                logging.error(f"StockDataset.split_train_test_dataset_by_stock() - Too few data, will be skipped. data shape: {raw_data.shape}")
                continue
            if self.predict_type.is_classify():#多分类
                dataset_y = self.get_binned_y_use_qcut(raw_y[:,[0,3]])  #只对t1l和t2h分箱,返回值为int类型
                raw_dataset_y = raw_y
            elif self.predict_type.is_binary():#二分类
                #按不同的二分类预测类型,生成对应的二分类y(0或1)
                if self.predict_type.is_t1_low():
                    dataset_y = (raw_y[:, 0]*100 <= self.predict_type.val).astype(int).reshape(-1, 1)
                elif self.predict_type.is_t1_high():
                    dataset_y = (raw_y[:, 1]*100 >= self.predict_type.val).astype(int).reshape(-1, 1)
                elif self.predict_type.is_t2_low():
                    dataset_y = (raw_y[:, 2]*100 <= self.predict_type.val).astype(int).reshape(-1, 1)
                elif self.predict_type.is_t2_high():
                    dataset_y = (raw_y[:, 3]*100 >= self.predict_type.val).astype(int).reshape(-1, 1)
                else:
                    raise ValueError(f"StockDataset.split_train_test_dataset_by_stock() - Unknown predict_type: {self.predict_type}")
                raw_dataset_y = raw_y
            elif self.predict_type.is_regression():#回归
                if self.predict_type.is_t1_low():
                    dataset_y = raw_y[:, 0].reshape(-1, 1).astype(float)*100
                elif self.predict_type.is_t1_high():
                    dataset_y = raw_y[:, 1].reshape(-1, 1).astype(float)*100
                elif self.predict_type.is_t2_low():
                    dataset_y = raw_y[:, 2].reshape(-1, 1).astype(float)*100
                elif self.predict_type.is_t2_high():
                    dataset_y = raw_y[:, 3].reshape(-1, 1).astype(float)*100
                # 裁剪极值，避免训练被极端样本主导
                dataset_y = np.clip(dataset_y, -CLIP_Y_PERCENT, CLIP_Y_PERCENT)

                raw_dataset_y = raw_y
            else:
                raise ValueError(f"StockDataset.split_train_test_dataset_by_stock() - Unknown predict_type: {self.predict_type}")
            train_count = int(len(raw_x) * train_size)
            test_count = len(raw_x) - train_count
            train_x_list.append(raw_x[test_count:])
            train_y_list.append(dataset_y[test_count:])
            raw_train_y_list.append(raw_dataset_y[test_count:])
            test_x_list.append(raw_x[:test_count])
            test_y_list.append(dataset_y[:test_count])
            raw_test_y_list.append(raw_dataset_y[:test_count])
            if collect_aligned_raw:
                ch_train_x_list.append(raw_data_with_idx[test_count:])
                ch_test_x_list.append(raw_data_with_idx[:test_count])

        # 合并所有股票的训练集和测试集
        raw_train_x = np.vstack(train_x_list) if train_x_list else np.array([])
        train_y = np.vstack(train_y_list) if train_y_list else np.array([])
        raw_train_y = np.vstack(raw_train_y_list) if raw_train_y_list else np.array([])
        raw_test_x = np.vstack(test_x_list) if test_x_list else np.array([])
        test_y = np.vstack(test_y_list) if test_y_list else np.array([])
        raw_test_y = np.vstack(raw_test_y_list) if raw_test_y_list else np.array([])
        #return (raw_train_x, train_y, raw_train_y), (raw_test_x, test_y, raw_test_y)
        #return (raw_train_x, train_y, raw_train_y), (raw_test_x, test_y, raw_test_y), aligned_raw_list  #for conv2d add
        if collect_aligned_raw:
            # 将通道切分结果存到实例属性，方便多通道流程使用
            self.channel_raw_data_list = aligned_raw_list
            self.channel_train_x_list = ch_train_x_list
            self.channel_test_x_list = ch_test_x_list
        return (raw_train_x, train_y, raw_train_y), (raw_test_x, test_y, raw_test_y)


    #归一化处理数据
    def get_normalized_data(self, raw_data):
        if raw_data is not None:
            df = pd.DataFrame(raw_data)
            #print(f"DEBUG:df head before norm:\n{df.head()}")
            scaled_data = self.scaler.transform(df) #归一化
        else:
            logging.error("Invalid parameters.")
            exit()
        return np.array(scaled_data).astype(float)
    
    #获取归一化参数
    def get_scaler(self, new_data=None, if_update=False, if_save=True):
        is_modified = False
        scaler_fn = os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + '_' + self.feature_type.value + '_scaler.save')

        if if_update:
            if new_data is not None or not os.path.exists(os.path.join(BASE_DIR, SCALER_DIR, self.stock.ts_code + "_scaler.save")):    #如果有输入数据参数或没有保存的归一化参数,则用该数据生成归一化参数
                self.scaler = StandardScaler()
                #self.scaler = RobustScaler()#
                #self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 替换RobustScaler
                if isinstance(new_data, np.ndarray) and new_data.ndim > 2:  #如果是多维数据,则先展开成二维
                    new_data = new_data.reshape(-1, new_data.shape[-1])
                df = pd.DataFrame(new_data)
                if df.isnull().values.any():
                    print("WARNING: 发现NaN, 归一化前已自动填充0")
                    df = df.fillna(0)
                if np.isinf(df.values.astype(float)).any():
                    print("WARNING: 发现inf, 归一化前已自动填充0")
                    df = df.replace([np.inf, -np.inf], 0)
                self.scaler.fit(df)
                is_modified = True
                if if_save and is_modified: #如果有更新且需要保存,则保存
                    joblib.dump(self.scaler, scaler_fn)
                    logging.info(f"write scaler cfg to {scaler_fn}")
            return self.scaler
        elif os.path.exists(scaler_fn):   #如果没有输入数据参数且有保存的归一化参数,则读取已有的归一化参数
            try:
                self.scaler = joblib.load(scaler_fn)
                logging.info(f"load scaler from {scaler_fn}")
                return self.scaler
            except Exception as e:
                logging.error(f"StockDataset.get_real_normalize() - Failed to load scaler: {e}")
                exit()
        else:
            logging.error(f"StockDataset.get_real_normalize() - scaler file not exists: {scaler_fn}")
            exit()
        

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
        """
        支持:
        - ndarray: 单通道旧逻辑 -> [样本, 时间步, 特征]
        - list of ndarray: 多通道，每个元素 shape [N, 全列(含日期/y)] -> 归一化后窗口化，再在最后一维堆 channel
        """
        self.get_scaler() if self.scaler is None else None
        if isinstance(raw_x, list): #多通道处理
            ch_windows = []
            for x in raw_x:
                feat = self._feature_only(x)
                norm = self.get_normalized_data(feat)
                win = self.get_windowed_x_by_raw(norm)
                ch_windows.append(win)
            # 对齐最短长度
            min_len = min(w.shape[0] for w in ch_windows)
            ch_windows = [w[:min_len] for w in ch_windows]
            x4d = np.stack(ch_windows, axis=-1)  # [样本, 时间步, 特征, channel]
            return x4d
        # 单通道保持原逻辑
        normalized_x = self.get_normalized_data(raw_x)
        return self.get_windowed_x_by_raw(normalized_x)
    
        if False:
            if self.use_conv2_channel:
                # raw_x: list，每个为(样本, features)
                # 对每个通道分别归一化、窗口化
                windowed_x = []
                for x_i in raw_x:
                    normed_x = self.get_normalized_data(x_i)
                    win_x = self.get_windowed_x_by_raw(normed_x)
                    windowed_x.append(win_x)
                # 长度对齐
                min_win = min([w.shape[0] for w in windowed_x])
                windowed_x = [w[:min_win] for w in windowed_x]
                x4d = np.stack(windowed_x, axis=-1)  # [样本, 时间步, feature, channel]
                return x4d
            else:
                self.get_scaler() if self.scaler is None else None  #如果还没有归一化参数,则先获取
                normalized_x = self.get_normalized_data(raw_x)
                return self.get_windowed_x_by_raw(normalized_x)
            
        

    #获取某一天的模型输入数据(已归一化),以及对应的真实结果y
    def get_dataset_with_y_by_date(self, date):
        date = type(self.raw_data[0, 0])(date)
        try:
            idx = np.where(self.date_list == date)[0][0]
        except Exception as e:
            logging.error(f"StockDataset.get_dataset_by_date() - Invalid date: {e}")
            exit()
        return self.normalized_windowed_train_x[idx,:,:] , self.raw_y[idx]
    
    #获取某一天的模型输入数据(已归一化, 预测用),以及对应的收盘价
    def get_predictable_dataset_by_date(self, date):
        if False:
            # ==== 新增: 支持多通道conv2模式 ====
            if self.use_conv2_channel:
                # 1. 主股票 + 所有关联股票为channels
                trades = [self.p_trade] + self.rel_trade_list
                idx_raw_data_list = [t.raw_data_np for t in self.idx_trade_list] if self.if_has_index else []
                x_windows = []
                min_win_len = None

                # 2. 遍历每个channel股票，找window并拼接指数
                for trade in trades:
                    #print(f"DEBUG: get_predictable_dataset_by_date() - processing trade: {trade.ts_code}")
                    data, _ = self.get_trade_data(trade)
                    date_val = type(data[0, 0])(date)
                    idx_arr = np.where(data[:, 0] == date_val)[0]
                    if idx_arr.size == 0 or idx_arr[0] + self.window_size > data.shape[0]:
                        raise ValueError(f"conv2 channel: {trade.ts_code} 日期{date}不可用")
                    idx0 = idx_arr[0]
                    seq = data[idx0: idx0 + self.window_size, :]
                    for idx_raw in idx_raw_data_list:
                        seq = self.left_join_pd_with_move_last(seq, idx_raw, move_last_n_cols=self.y_cnt)
                    seq_x, _ = self.get_dataset_xy(seq)
                    x_windows.append(seq_x)
                    if min_win_len is None or seq_x.shape[0] < min_win_len:
                        min_win_len = seq_x.shape[0]
                x_windows = [x[:min_win_len] for x in x_windows]
                x_windows = [self.get_normalized_data(x) for x in x_windows]
                x4d = np.stack(x_windows, axis=-1)      # [window, feature, channel]
                x4d = np.expand_dims(x4d, axis=0)       # [1, window, feature, channel]
                # 取主股票的close为基准价
                closed_price = None
                main_data, _ = self.get_trade_data(self.p_trade)
                main_idx_arr = np.where(main_data[:, 0] == date_val)[0]
                if main_idx_arr.size:
                    closed_price = main_data[main_idx_arr[0], self.p_trade.col_close + 1]
                #print(f"DEBUG: get_predictable_dataset_by_date() - date={date}, x4d.shape={x4d.shape}")
                return x4d, closed_price
            
            # ==== 旧版: 单通道模式 ====
            date = self.si.get_next_or_current_trade_date(date) #若输入日期为交易日,则返回该日期,否则返回后一个交易日
            date = type(self.raw_data[0, 0])(date)
            try:
                idx = np.where(self.raw_data[:, 0] == date)[0][0]
            except Exception as e:
                logging.error(f"StockDataset.get_predictable_dataset_by_date() - Invalid date: {e}")
                exit()
            if idx + self.window_size > self.raw_data.shape[0]:
                raise ValueError(f"Not enough data for window: idx={idx}, window_size={self.window_size}, data_len={self.raw_data.shape[0]}")
            closed_price = self.raw_data[idx, self.p_trade.col_close + 1] #取出对应日期的收盘价, +1是因为raw_data含日期列
            raw_x = self.raw_data[idx : idx + self.window_size, 1:]#取出对应日期及之后window_size天的数据
            x = self.get_normalized_windowed_x(raw_x) #归一化,窗口化
            #print(f"DEBUG: date={date}, idx={idx}, raw_x.shape={raw_x.shape}, x.shape={x.shape}, closed_price={closed_price}")
            return x, closed_price
        
        date = self.si.get_next_or_current_trade_date(date) #若输入日期为交易日,则返回该日期,否则返回后一个交易日
        date_val = type(self.raw_data[0, 0])(date)

        # 多通道：逐通道取窗口 -> 归一化 -> stack channel
        if self.use_conv2_channel and getattr(self, "channel_raw_data_list", None):
            x_list = []
            main_close = None
            for i, ch_raw in enumerate(self.channel_raw_data_list):
                idx_arr = np.where(ch_raw[:, 0] == date_val)[0]
                if idx_arr.size == 0 or idx_arr[0] + self.window_size > ch_raw.shape[0]:
                    raise ValueError(f"conv2 channel: 日期{date}不可用或长度不足, channel={i}")
                idx0 = idx_arr[0]
                if i == 0:
                    main_close = ch_raw[idx0, self.p_trade.col_close + 1]
                seq = ch_raw[idx0: idx0 + self.window_size, :]
                seq_feat = self._feature_only(seq)
                norm = self.get_normalized_data(seq_feat)
                x_list.append(norm)
            min_len = min(w.shape[0] for w in x_list)
            x_list = [w[:min_len] for w in x_list]
            x4d = np.stack(x_list, axis=-1)  # [window, feature, channel]
            x4d = np.expand_dims(x4d, axis=0)  # [1, window, feature, channel]
            return x4d, main_close

        # 单通道保持原逻辑
        try:
            idx = np.where(self.raw_data[:, 0] == date_val)[0][0]
        except Exception as e:
            logging.error(f"StockDataset.get_predictable_dataset_by_date() - Invalid date: {e}")
            exit()
        if idx + self.window_size > self.raw_data.shape[0]:
            raise ValueError(f"Not enough data for window: idx={idx}, window_size={self.window_size}, data_len={self.raw_data.shape[0]}")
        closed_price = self.raw_data[idx, self.p_trade.col_close + 1] #取出对应日期的收盘价, +1是因为raw_data含日期列
        raw_x = self.raw_data[idx : idx + self.window_size, 1:]#取出对应日期及之后window_size天的数据
        x = self.get_normalized_windowed_x(raw_x) #归一化,窗口化
        return x, closed_price


    def left_join_pd_with_move_last(self, A, B, move_last_n_cols=0, if_debug=False):
        """
        用 pandas 实现 left join，并将A的倒数N列移至结果的倒数N列（可控）
        A: shape (n, m1)
        B: shape (k, m2)
        move_last_n_cols: int，A的倒数N列移到最终结果最后N列
        返回: shape (n, m1 + m2 - 1)
        """
        #print(f"typeof A: {type(A)}, shape: {getattr(A,'shape',None)}")
        if not (isinstance(A, np.ndarray) and A.ndim == 2):
            raise ValueError(f"A must be 2D np.ndarray, got {type(A)}, shape={getattr(A,'shape',None)}")
        if not (isinstance(B, np.ndarray) and B.ndim == 2):
            raise ValueError(f"B must be 2D np.ndarray, got {type(B)}, shape={getattr(B,'shape',None)}")
        df_a = pd.DataFrame(A)
        df_b = pd.DataFrame(B)
        if if_debug:
            print(f"before datetime conversion:"+"*"*40)
            print(f"df_a head:\n{df_a.head(3)}\ndf_b head:\n{df_b.head(3)}")
        # 保证第0列类型一致
        df_a[0] = pd.to_datetime(df_a[0]).dt.strftime('%Y%m%d')
        df_b[0] = pd.to_datetime(df_b[0]).dt.strftime('%Y%m%d')
        if if_debug:
            print(f"after datetime conversion:"+"*"*40)
            print(f"df_a head:\n{df_a.head(3)}\ndf_b head:\n{df_b.head(3)}")
        # 合并，左连接，按第0列key
        df_merged = pd.merge(df_a, df_b, left_on=0, right_on=0, how='left', suffixes=('', '_b'))
        if if_debug:
            print(f"after merge:"+"*"*40)
            print(f"df_merged head:\n{df_merged.head(3)}")
        # 去掉B重复的key列
        df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

        if move_last_n_cols > 0:
            orig_cols = df_a.shape[1]
            total_cols = df_merged.shape[1]
            # A的倒数N列的索引
            idx_move = list(range(orig_cols - move_last_n_cols, orig_cols))
            # 其他A列（不含倒数N列）
            idx_keep = list(range(0, orig_cols - move_last_n_cols))
            # B部分的列索引（合并后，除了A的全部列）
            idx_rest = list(range(orig_cols, total_cols))
            # 新顺序 = A的前面 + B部分 + A倒数N列
            new_idx = idx_keep + idx_rest + idx_move
            df_merged = df_merged.iloc[:, new_idx]
        #print(f"typeof df_merged: {type(df_merged)}, shape: {getattr(df_merged,'shape',None)}")
        return df_merged.values
    
    def get_feature_names(self):
        """
        返回主股票和指数合并后的特征名列表（不包含日期、ts_code）。
        """
        feature_names = []
        feature_names += list(self.p_trade.trade_df.columns.drop(['ts_code','trade_date'], errors='ignore'))
        for idx_trade in getattr(self, 'idx_trade_list', []):
            feature_names += list(idx_trade.trade_df.columns.drop(['ts_code','trade_date'], errors='ignore'))
        return feature_names

    def time_series_augmentation(self, x_data, y_data, noise_level=0.01):
        """
        通过添加微小的高斯噪声进行时间序列数据增强
        """
        augmented_x = x_data.copy()
        # 添加高斯噪声
        noise = np.random.normal(0, noise_level, augmented_x.shape)
        augmented_x = augmented_x + noise
        return augmented_x, y_data

    def time_series_augmentation_multiple(self, X, y, multiple=4, noise_level=0.005):
        """
        时间序列数据增强，包含多种增强方法
        """
        X_aug = X.copy()
        y_aug = y.copy()
        
        # 1. 添加高斯噪声
        if multiple >= 2:
            X_noise = X + np.random.normal(0, noise_level, X.shape)
            X_aug = np.concatenate([X_aug, X_noise])
            y_aug = np.concatenate([y_aug, y])
        
        # 2. 时间扭曲 (Time Warping)
        if multiple >= 3:
            X_warp = []
            for i in range(X.shape[0]):
                if i > 0 and i < X.shape[0] - 1:
                    warped = 0.5 * X[i-1] + 0.5 * X[i]
                    X_warp.append(warped)
            if X_warp:
                X_warp = np.array(X_warp)
                y_warp = y[:len(X_warp)]
                X_aug = np.concatenate([X_aug, X_warp])
                y_aug = np.concatenate([y_aug, y_warp])
        
        # 3. 缩放变换
        if multiple >= 4:
            scale_factor = np.random.uniform(0.95, 1.05)
            X_scaled = X * scale_factor
            X_aug = np.concatenate([X_aug, X_scaled])
            y_aug = np.concatenate([y_aug, y])
        
        return X_aug, y_aug

    #根据日期获取原始y数据(4个值)
    def get_raw_y_by_date(self, date):
        date = type(self.raw_data[0, 0])(date)
        try:
            idx = np.where(self.date_list == date)[0][0]
        except Exception as e:
            logging.error(f"StockDataset.get_raw_y_by_date() - Invalid date: {e}")
            exit()
        return self.raw_y[idx]
    
    def get_real_y_by_raw_y(self, raw_y):
        if self.predict_type.is_classify():#多分类
            return self.get_binned_y_use_qcut(raw_y)
        elif self.predict_type.is_binary():#二分类
            #按不同的二分类预测类型,生成对应的二分类y(0或1)
            if self.predict_type.is_t1_low():
                return (raw_y[:,0]*100 <= self.predict_type.val).astype(int).reshape(-1, 1)
            elif self.predict_type.is_t1_high():
                return (raw_y[:,1]*100 >= self.predict_type.val).astype(int).reshape(-1, 1)
            elif self.predict_type.is_t2_low():
                return (raw_y[:,2]*100 <= self.predict_type.val).astype(int).reshape(-1, 1)
            elif self.predict_type.is_t2_high():
                return (raw_y[:,3]*100 >= self.predict_type.val).astype(int).reshape(-1, 1)
            else:
                raise ValueError(f"StockDataset.get_real_y_by_raw_y() - Unknown predict_type: {self.predict_type}")
        elif self.predict_type.is_regression():#回归
            if self.predict_type.is_t1_low():
                return raw_y[:,0].reshape(-1,1).astype(float)
            elif self.predict_type.is_t1_high():
                return raw_y[:,1].reshape(-1,1).astype(float)
            elif self.predict_type.is_t2_low():
                return raw_y[:,2].reshape(-1,1).astype(float)
            elif self.predict_type.is_t2_high():
                return raw_y[:,3].reshape(-1,1).astype(float)
            else:
                raise ValueError(f"StockDataset.get_real_y_by_raw_y() - Unknown predict_type: {self.predict_type}")            
        else:
            raise ValueError(f"StockDataset.get_real_y_by_raw_y() - Unknown predict_type: {self.predict_type}")

    #根据日期获取真实y数据(1或2个值)
    def get_real_y_by_date(self, date):
        date = type(self.raw_data[0, 0])(date)
        raw_y = self.get_raw_y_by_date(date).reshape(1,-1)
        real_y = self.get_real_y_by_raw_y(raw_y)
        return real_y[0,0] if real_y.shape[1]==1 else real_y[0]
        
    # ---------- 工具：仅保留特征列、去掉日期和y ----------
    def _feature_only(self, x):
        return x[:, 1:-self.y_cnt] if self.y_cnt > 0 else x[:, 1:]

    def _stack_features_for_scaler(self, raw_train_x):
        """
        保证传入 scaler 的都是纯特征:
        - ndarray: 去掉日期、y
        - list: 各通道去掉日期、y 后 vstack
        """
        if isinstance(raw_train_x, list):
            feats = [self._feature_only(arr) for arr in raw_train_x]
            return np.vstack(feats)
        else:
            return raw_train_x    

if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    #download_list = si.get_filtered_stock_list(mmv=3000000)
    primary_stock_code = '600036.SH'
    idx_code_list = IDX_CODE_LIST
    rel_code_list = ALL_CODE_LIST#ALL_CODE_LIST#BANK_CODE_LIST
    #ds = StockDataset(primary_stock_code, idx_code_list, rel_code_list, si, start_date='19910104', end_date='20250903', train_size=0.8)
    #ds = StockDataset(primary_stock_code, idx_code_list, rel_code_list, si, start_date='20190104', end_date='20250903', 
    #                  train_size=0.9, if_use_all_features=False, predict_type=PredictType.BINARY_T2_L10)

    #ds = StockDataset(ts_code=primary_stock_code, idx_code_list=idx_code_list, rel_code_list=[], si=si, if_update_scaler=False,
    #            start_date='19921203', end_date='20250930',
    #            train_size=1, feature_type=FeatureType.T1L10_F55, predict_type=PredictType.BINARY_T1_L10)

    ds = StockDataset(
        ts_code=primary_stock_code,
        idx_code_list=idx_code_list,
        rel_code_list=rel_code_list,
        si=si,
        start_date='20150701',
        end_date='20251201',
        train_size=1,
        feature_type=FeatureType.BINARY_T1L10_F55,
        if_update_scaler=True,
        predict_type=PredictType.BINARY_T1_L10,
        use_conv2_channel=True,
    )

    logging.info(f"ds.train_y shape: {ds.train_y.shape}, ds.test_y shape: {ds.test_y.shape}")
    pd.set_option('display.max_columns', None)
    start_idx = 0
    print(f"\nraw x sample: \n{pd.DataFrame(ds.raw_data).iloc[start_idx:start_idx+3]}")
    print(f"\nraw y sample: \n{pd.DataFrame(ds.raw_y).iloc[start_idx:start_idx+3]}")
    #print(f"feature names: {ds.get_feature_names()}")
    #data, bp = ds.get_predictable_dataset_by_date("20250829")
    #print(f"data shape: {data.shape}, bp: {bp}")
    #print(f"{ds.p_trade.remain_list}")
    #print(f"data: \n{pd.DataFrame(data[0])}")
