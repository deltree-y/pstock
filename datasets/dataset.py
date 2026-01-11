#-*- coding:UTF-8 -*-
import sys,os,time,logging,joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Optional, List   # <<< ADD
from stockinfo import StockInfo
from trade import Trade
from cat import RateCat
from bins import BinManager
from utils.tk import TOKEN
from utils.utils import FeatureType, setup_logging
from utils.utils import StockType, PredictType
from utils.const_def import CONTINUOUS_DAYS, NUM_CLASSES, MIN_TRADE_DATA_ROWS, T1L_SCALE, T2H_SCALE, ALL_CODE_LIST, IDX_CODE_LIST, CLIP_Y_PERCENT, CODE_LIST_TEMP, BANK_CODE_LIST_10
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
    def __init__(self, ts_code, idx_code_list, rel_code_list, si:StockInfo, start_date=None, end_date=None, train_size=0.8, if_update_scaler=True, 
                 feature_type:FeatureType=FeatureType.ALL, predict_type:PredictType=PredictType.CLASSIFY, use_conv2_channel=False,
                 custom_feature_list: Optional[List[str]] = None,   # <<< ADD
                ):
        self.p_trade = Trade(ts_code, si,
                             start_date=start_date, end_date=end_date, 
                             feature_type=feature_type,
                             custom_feature_list=custom_feature_list,        # <<< ADD
                            )
        self.idx_trade_list = [
            Trade(idx_code, si, stock_type=StockType.INDEX, start_date=start_date, end_date=end_date, feature_type=feature_type) 
            for idx_code in idx_code_list
            ]
        self.rel_trade_list = []
        for rel_code in rel_code_list:
            t = Trade(
                rel_code, si,
                stock_type=StockType.RELATED,
                start_date=start_date, end_date=end_date,
                feature_type=feature_type,
                custom_feature_list=custom_feature_list,    # <<< ADD：保证 rel 与主股输入维度一致
            )
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

        # 1. 预处理输入数据
        #处理主股票数据
        self.datasets, self.raw_data = self.get_trade_data(self.p_trade)  #datasets包含y不含t1t2数据(第一列为日期), raw_data包含t1t2数据不含y
        self.datasets_date_list = self.datasets[:,0]     #不含t1,t2数据的日期列表
        self.raw_date_list = self.raw_data[:,0]          #含t1,t2数据的日期列表
        #处理关联股票数据(如果是多通道处理, 则不进行关联股票数据合并)
        self.rel_datasets_list, self.rel_raw_data_list = zip(*[self.get_trade_data(rel_trade) for rel_trade in self.rel_trade_list]) if self.if_has_related else ([], [])
        if not self.use_conv2_channel:  #多通道处理时, 不进行关联股票数据合并
            self.raw_data = np.vstack(([self.raw_data] + list(self.rel_raw_data_list)) if self.if_has_related else self.raw_data)   #已包括主股票及关联股票数据
        #处理指数数据(将指数数据并接到主股票及关联股票数据(raw_data)上)
        self.idx_datasets_list, self.idx_raw_data_list = zip(*[self.get_trade_data(idx_trade) for idx_trade in self.idx_trade_list]) if self.if_has_index else ([], [])
        if self.if_has_index:
            #for idx_raw_data in self.idx_raw_data_list: #逐个将指数数据并接到主股票及关联股票数据(raw_data)上, 如果批量处理的话, 会打乱已有的raw_data顺序
            #    self.raw_data = self.left_join_pd_with_move_last(self.raw_data, idx_raw_data, if_debug=False) if self.if_has_index else self.raw_data    #已包括主股票及关联股票及指数数据
            for idx_trade, idx_raw_data in zip(self.idx_trade_list, self.idx_raw_data_list):
                self.raw_data = self._left_join_idx_with_prefix(
                    A_np=self.raw_data,
                    B_np=idx_raw_data,
                    idx_ts_code=idx_trade.ts_code,
                    B_trade_df_cols=list(idx_trade.trade_df.columns),  # ['ts_code','trade_date',...features...]
                    move_last_n_cols=0,  # raw_data_np 不含 y，所以不需要 move_last
                    if_debug=False
                )
        #self._print_channel_lengths() if len(rel_code_list)>0 else None  # Debug: 打印各通道长度 & 最短通道 ts_code

        # 2. 合并主股票与相关联股票的原始数据(如果是多通道数据,则再进行一次按日期对齐处理,确保各通道数据日期与主股票一致)
        datasets_list = [self.datasets] + list(self.rel_datasets_list) if self.if_has_related  else [self.datasets]
        if self.use_conv2_channel:
            datasets_list = self.align_channels_with_fill(datasets_list, fill_val=-1)  #多通道处理时, 按主股票的日期对齐各通道数据

        # 2.5 基于所有的原始y数据生成分箱器
        if self.use_conv2_channel:  #多通道处理时, 只用主股票数据生成分箱器
            self.raw_y = self.get_y_from_datasets(datasets_list[0]) #取出y(多个)
        else:   #非多通道处理时, 用所有股票数据生成分箱器
            self.raw_y = self.get_y_from_datasets(np.vstack(datasets_list)) #取出y(多个)
        self.bins1, self.bins2 = self.get_bins(self.raw_y)

        # 3. 按股票分离测试集与验证集,并对y进行分箱,返回对应的y
        (self.raw_train_x, self.train_y, self.raw_train_y), (self.raw_test_x, self.test_y, self.raw_test_y) = \
            self.split_train_test_dataset_by_stock(datasets_list, self.train_size) 

        # 4. 根据train_x的数据,生成并保存\读取归一化参数, #根据输入参数判断是否需要更新归一化参数配置,如果更新的话,就保存新的参数配置
        self.scaler = self.get_scaler(
            new_data=self._stack_features_for_scaler(self.raw_train_x),
            if_update=self.if_update_scaler,
            if_save=True
        )

        # 5. 归一化处理train_x/test_x, 并对x窗口化
        self.normalized_windowed_train_x = self.get_normalized_windowed_x(self.raw_train_x)
        self.normalized_windowed_test_x = self.get_normalized_windowed_x(self.raw_test_x) if train_size < 1 else None

        # 5.5 保存未归一化的x数据,供有需要的使用
        self.raw_windowed_train_x = self.get_raw_windowed_x(self.raw_train_x)
        self.raw_windowed_test_x = self.get_raw_windowed_x(self.raw_test_x) if train_size < 1 else None
        
        # 6. 对齐y数据, 因为x窗口化后会减少数据,所以y也要按窗口大小相应减少
        self.train_y_no_window, self.test_y_no_window = self.train_y, self.test_y #保存未窗口化的y数据,供有需要的使用
        self.train_y, self.test_y = self.train_y[self.window_size-1:], self.test_y[self.window_size-1:] #由于x按窗口化后会减少数据,所以y也要相应减少
        self.raw_train_y, self.raw_test_y = self.raw_train_y[self.window_size-1:], self.raw_test_y[self.window_size-1:] #由于x按窗口化后会减少数据,所以y也要相应减少
        #self.train_y, self.test_y = self.train_y[:-self.window_size+1], self.test_y[:-self.window_size+1] #由于x按窗口化后会减少数据,所以y也要相应减少

        #logging.info(f"train x/y shape - <{self.normalized_windowed_train_x.shape}/{self.train_y.shape}>")
        #logging.info(f"test  x/y shape - <{self.normalized_windowed_test_x.shape}/{self.test_y.shape}>")

    #所有从trade获取的数据都由此方法返回
    def get_trade_data(self, trade):
        return trade.combine_data_np, trade.raw_data_np
    
    #从原始数据集中取出y数据
    def get_y_from_datasets(self, raw_dataset):
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
    def split_train_test_dataset_by_stock(self, datasets_list, train_size):
        train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
        raw_train_y_list, raw_test_y_list = [], [] #保存未分箱的y数据,供有需要的使用
        aligned_raw_list = [] if self.use_conv2_channel else None  #for conv2d add
        ch_train_x_list, ch_test_x_list = [], []  # 每通道切分后的X（含日期+y，后续再裁剪特征）
        for datasets in datasets_list:  #逐只股票处理. 若为多通道处理,则需要进行日期对齐处理
            datasets_with_idx = datasets
            if self.if_has_index:   #并接指数数据
                #for idx_raw_data in self.idx_raw_data_list:
                #    datasets_with_idx = self.left_join_pd_with_move_last(datasets_with_idx, idx_raw_data, move_last_n_cols=self.y_cnt) if self.if_has_index else datasets    #并接指数数据
                for idx_trade, idx_raw_data in zip(self.idx_trade_list, self.idx_raw_data_list):
                    datasets_with_idx = self._left_join_idx_with_prefix(
                        A_np=datasets_with_idx,
                        B_np=idx_raw_data,
                        idx_ts_code=idx_trade.ts_code,
                        B_trade_df_cols=list(idx_trade.trade_df.columns),
                        move_last_n_cols=self.y_cnt,
                        if_debug=False
                    )
            if self.use_conv2_channel:                      #for conv2d add
                aligned_raw_list.append(datasets_with_idx)  #for conv2d add #aligned_raw_list有什么用???
            dataset_x, raw_y = self.get_dataset_xy(datasets_with_idx)   #raw_y为原始y涨跌幅数据,float类型
            if len(dataset_x) < NUM_CLASSES:
                logging.error(f"StockDataset.split_train_test_dataset_by_stock() - Too few data, will be skipped. data shape: {datasets.shape}")
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
                    dataset_y = raw_y[:, 0].reshape(-1, 1).astype(float) * 100
                elif self.predict_type.is_t1_high():
                    dataset_y = raw_y[:, 1].reshape(-1, 1).astype(float) * 100
                elif self.predict_type.is_t2_low():
                    dataset_y = raw_y[:, 2].reshape(-1, 1).astype(float) * 100
                elif self.predict_type.is_t2_high():
                    dataset_y = raw_y[:, 3].reshape(-1, 1).astype(float) * 100
                # 裁剪极值，避免训练被极端样本主导
                dataset_y = np.clip(dataset_y, -CLIP_Y_PERCENT, CLIP_Y_PERCENT)
                raw_dataset_y = raw_y
            else:
                raise ValueError(f"StockDataset.split_train_test_dataset_by_stock() - Unknown predict_type: {self.predict_type}")

            train_count, test_count = int(len(dataset_x) * train_size), len(dataset_x) - int(len(dataset_x) * train_size) 
            train_x_list.append(dataset_x[:train_count])
            train_y_list.append(dataset_y[:train_count])
            raw_train_y_list.append(raw_dataset_y[:train_count])
            test_x_list.append(dataset_x[train_count:])
            test_y_list.append(dataset_y[train_count:])
            raw_test_y_list.append(raw_dataset_y[train_count:])
            if self.use_conv2_channel:  #for conv2d add
                ch_train_x_list.append(datasets_with_idx[:train_count])
                ch_test_x_list.append(datasets_with_idx[train_count:])

        # 合并所有股票的训练集和测试集
        train_x = np.vstack(train_x_list) if not self.use_conv2_channel else np.array(train_x_list)
        train_y = np.vstack(train_y_list) if not self.use_conv2_channel else np.array(train_y_list[0])              #多通道时, y只取主股票的
        raw_train_y = np.vstack(raw_train_y_list) if not self.use_conv2_channel else np.array(raw_train_y_list[0])  #多通道时, y只取主股票的
        test_x = np.vstack(test_x_list) if not self.use_conv2_channel else np.array(test_x_list)
        test_y = np.vstack(test_y_list) if not self.use_conv2_channel else np.array(test_y_list[0])                 #多通道时, y只取主股票的
        raw_test_y = np.vstack(raw_test_y_list) if not self.use_conv2_channel else np.array(raw_test_y_list[0])     #多通道时, y只取主股票的

        if self.use_conv2_channel:  ############# for conv2d add
            # 将通道切分结果存到实例属性，方便多通道流程使用
            self.channel_raw_data_list = aligned_raw_list
            self.channel_train_x_list = ch_train_x_list
            self.channel_test_x_list = ch_test_x_list
        return (train_x, train_y, raw_train_y), (test_x, test_y, raw_test_y)


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
        

    # 将输入的二维x,按窗口大小处理成三维的数据
    # 升序(old->new)下：每个窗口为 [t-(win-1) ... t]，即窗口最后一天是 T0
    # 输出x格式 - [样本, 时间步, 特征]
    def get_windowed_x_by_raw(self, dataset_x=None):
        if dataset_x is None:
            raise ValueError("dataset_x is None")
        x = []
        n = len(dataset_x)
        w = self.window_size
        # i 表示窗口起点，窗口覆盖 [i, i+w)
        for i in range(0, n - w + 1):
            x.append(dataset_x[i:i+w])
        return np.asarray(x, dtype=float)
        
    #获取归一化,窗口化后的x数据
    def get_normalized_windowed_x(self, raw_x:np.array, if_normalize=True):
        """
        支持:
        - ndarray: 单通道旧逻辑 -> [样本, 时间步, 特征]
        - list of ndarray: 多通道，每个元素 shape [N, 全列(含日期/y)] -> 归一化后窗口化，再在最后一维堆 channel
        """
        self.get_scaler() if self.scaler is None else None  #确保已加载归一化参数
        if raw_x.ndim == 3: #多通道处理
            ch_windows = []
            for x in raw_x:
                norm = self.get_normalized_data(x) if if_normalize else x
                win = self.get_windowed_x_by_raw(norm) 
                ch_windows.append(win)
            x4d = np.stack(ch_windows, axis=-1)  # [样本, 时间步, 特征, channel]
            return x4d
        # 单通道保持原逻辑
        normalized_x = self.get_normalized_data(raw_x) if if_normalize else raw_x
        return self.get_windowed_x_by_raw(normalized_x)
    
    #获取未归一化,窗口化后的x数据
    def get_raw_windowed_x(self, raw_x:np.array):
        return self.get_normalized_windowed_x(raw_x, if_normalize=False)

    #获取某一天的模型输入数据(已归一化),以及对应的真实结果y
    def get_dataset_with_y_by_date(self, date):
        date = type(self.raw_data[0, 0])(date)
        try:
            idx = np.where(self.datasets_date_list == date)[0][0]
        except Exception as e:
            logging.error(f"StockDataset.get_dataset_by_date() - Invalid date: {e}")
            exit()
        return self.normalized_windowed_train_x[idx,:,:] , self.raw_y[idx]
    
    #获取某一天的模型输入数据(已归一化, 预测用),以及对应的收盘价
    def get_predictable_dataset_by_date(self, date):
        date = self.si.get_next_or_current_trade_date(date) #若输入日期为交易日,则返回该日期,否则返回后一个交易日
        date_val = type(self.raw_data[0, 0])(date)

        # 多通道：逐通道取窗口 -> 归一化 -> stack channel
        if self.use_conv2_channel and getattr(self, "channel_raw_data_list", None):
            x_list = []
            main_close = None
            for i, ch_raw in enumerate(self.channel_raw_data_list):
                idx_arr = np.where(ch_raw[:, 0] == date_val)[0] #找到对应日期的索引位置
                if idx_arr.size == 0:
                    raise ValueError(f"conv2 channel: 日期{date}不可用, channel={i}")

                idx0 = int(idx_arr[0])
                start = idx0 - self.window_size + 1
                end = idx0 + 1  # python slice right-open

                # 历史窗口不足
                if start < 0:
                    raise ValueError(
                        f"conv2 channel: Not enough history for window: "
                        f"idx={idx0}, window_size={self.window_size}, data_len={ch_raw.shape[0]}, channel={i}"
                    )

                if i == 0:
                    main_close = ch_raw[idx0, self.p_trade.col_close + 1]  # T0 close

                seq = ch_raw[start:end, :]          # [window, all_cols]
                seq_feat = self._feature_only(seq)  # drop date & y
                norm = self.get_normalized_data(seq_feat)

                # 这里 norm 已经是一个窗口，不需要再滑窗；直接 expand 成 [1, window, feature]
                win = np.expand_dims(norm, axis=0)
                x_list.append(win[0])  # [window, feature]

            x4d = np.stack(x_list, axis=-1)     # [window, feature, channel]
            x4d = np.expand_dims(x4d, axis=0)   # [1, window, feature, channel]
            return x4d, float(main_close)

        # ===== 单通道：取“历史窗口” =====
        idx_arr = np.where(self.raw_data[:, 0] == date_val)[0]
        if idx_arr.size == 0:
            raise ValueError(f"StockDataset.get_predictable_dataset_by_date() - Invalid date: {date}, last date in data: {self.raw_data[-1,0]}")
        idx = int(idx_arr[0])
        start = idx - self.window_size + 1
        end = idx + 1
        if start < 0:
            raise ValueError(
                f"Not enough history for window: idx={idx}, window_size={self.window_size}, data_len={self.raw_data.shape[0]}"
            )
        closed_price = float(self.raw_data[idx, self.p_trade.col_close + 1])  # T0 close, +1是因为raw_data含日期列
        raw_x = self.raw_data[start:end, 1:]  # 取过去 window_size 天(含当天)的特征

        # raw_x 已经是窗口，get_normalized_windowed_x 会再滑窗一次（得到1个样本），符合你原来的输出形状
        x = self.get_normalized_windowed_x(raw_x)
        return raw_x, x, closed_price

    def _left_join_idx_with_prefix(self, A_np, B_np, idx_ts_code: str, B_trade_df_cols, move_last_n_cols=0, if_debug=False):
        """
        将指数/外部序列 B left join 到 A 上，按 trade_date 对齐，并对 B 的特征列加前缀:
            idx_{ts_code}__{col}

        输入约定：
        - A_np: shape [n, mA]，第0列为 trade_date（string/int均可），其余为数值（含特征、可选y）
        - B_np: shape [k, mB]，第0列为 trade_date（string/int均可），其余为数值（只应是特征；idx 本身没有 y）
        - B_trade_df_cols: 对应 Trade.trade_df.columns（含 ts_code/trade_date/...）
                          用它恢复 B_np 的列名（但 B_np 本身不含 ts_code，因此我们会 drop 掉 ts_code）
        - move_last_n_cols: 若 A 的末尾有 y 列（如 datasets_with_idx），传 y_cnt 用于把 y 移回最后

        返回：
        - np.ndarray: shape [n, mA + (mB-1)] （按 trade_date 合并，B 不重复加入 trade_date）
        """
        if not (isinstance(A_np, np.ndarray) and A_np.ndim == 2):
            raise ValueError(f"A_np must be 2D np.ndarray, got {type(A_np)}, shape={getattr(A_np,'shape',None)}")
        if not (isinstance(B_np, np.ndarray) and B_np.ndim == 2):
            raise ValueError(f"B_np must be 2D np.ndarray, got {type(B_np)}, shape={getattr(B_np,'shape',None)}")

        # ---- build df_a ----
        df_a = pd.DataFrame(A_np).copy()

        # ---- build df_b with names ----
        # B_np 结构是: [trade_date] + feature...
        # idx_trade.trade_df.columns 结构是: ['ts_code','trade_date', feature...]
        # 所以我们要 drop ts_code 后得到: ['trade_date', feature...]
        b_cols = [c for c in B_trade_df_cols if c != 'ts_code']
        if len(b_cols) != B_np.shape[1]:
            raise ValueError(
                f"idx cols mismatch for {idx_ts_code}: "
                f"len(b_cols)={len(b_cols)} vs B_np.shape[1]={B_np.shape[1]}. "
                f"b_cols(head)={b_cols[:5]}"
            )

        df_b = pd.DataFrame(B_np, columns=b_cols).copy()

        # ---- normalize key dtype ----
        df_a[0] = pd.to_datetime(df_a[0]).dt.strftime('%Y%m%d')
        df_b['trade_date'] = pd.to_datetime(df_b['trade_date']).dt.strftime('%Y%m%d')

        # ---- prefix B feature columns (exclude key) ----
        pref = f"idx_{idx_ts_code}__"
        rename_map = {c: (pref + c) for c in df_b.columns if c != 'trade_date'}
        df_b.rename(columns=rename_map, inplace=True)

        if if_debug:
            print(f"[DEBUG] idx join: idx_ts_code={idx_ts_code}, renamed cols sample: {list(rename_map.items())[:5]}")

        # ---- merge (left join on trade_date) ----
        # df_a 的 key 是第0列（整数列名），df_b 的 key 是 'trade_date'
        df_merged = pd.merge(df_a, df_b, left_on=0, right_on='trade_date', how='left')

        # 删除 df_b 的 trade_date（避免重复）
        df_merged.drop(columns=['trade_date'], inplace=True)

        # ---- move last N cols (keep y at end) ----
        if move_last_n_cols and move_last_n_cols > 0:
            orig_cols = df_a.shape[1]
            total_cols = df_merged.shape[1]

            idx_move = list(range(orig_cols - move_last_n_cols, orig_cols))
            idx_keep = list(range(0, orig_cols - move_last_n_cols))
            idx_rest = list(range(orig_cols, total_cols))
            new_idx = idx_keep + idx_rest + idx_move
            df_merged = df_merged.iloc[:, new_idx]

        return df_merged.values

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
        返回最终输入模型的特征名列表（不包含日期、ts_code、y）。
        注意：related 股票在 use_conv2_channel=False 时是 vstack 到样本维，不会新增列名；
              idx 股票是拼到列维，因此会新增带前缀的列。
        """
        feature_names = []
        feature_names += list(self.p_trade.trade_df.columns.drop(['ts_code','trade_date'], errors='ignore'))

        for idx_trade in getattr(self, 'idx_trade_list', []):
            pref = f"idx_{idx_trade.ts_code}__"
            idx_cols = list(idx_trade.trade_df.columns.drop(['ts_code','trade_date'], errors='ignore'))
            feature_names += [pref + c for c in idx_cols]

        return feature_names

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
            idx = np.where(self.datasets_date_list == date)[0][0]
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
        if raw_train_x.ndim == 3:  #多通道处理
            return np.vstack(raw_train_x)
        else:   #单通道处理
            return raw_train_x    
    
    def align_channels_with_fill(self, channel_data_list, fill_val=-1):
        """
        多通道对齐填充。
        channel_data_list: list of np.ndarray，每个 shape [N, feature]，首列为日期，[0]为主股票
        fill_val: 缺失填充值
        返回：list of ndarray，各通道 shape=[主股票 N, feature]
        """
        main_data = channel_data_list[0]
        main_dates = main_data[:, 0]
        feature_dim_list = [arr.shape[1] - 1 for arr in channel_data_list]  # 不包含日期列
        aligned_channels = []
        for ch, feat_dim in zip(channel_data_list, feature_dim_list):
            # 生成 {date: row数据（不含date）}
            ch_dict = {int(row[0]): row[1:] for row in ch}
            arr = np.vstack([
                ch_dict.get(int(date), np.full(feat_dim, fill_val, dtype=float))
                for date in main_dates
            ])
            # 保留主股票原始日期列，其它参考通道用对齐后的日期
            arr = np.column_stack([main_dates, arr])
            aligned_channels.append(arr)
            #print(f"DEBUG: align channel, original len: {ch.shape[0]}, aligned len: {arr.shape[0]}")
        return aligned_channels
    
    # ===== Debug: 打印各通道长度 & 最短通道 ts_code =====        
    def _print_channel_lengths(self):
        channel_trades = [self.p_trade] + list(self.rel_trade_list)
        channel_lens = [t.combine_data_np.shape[0] for t in channel_trades]
        min_len = min(channel_lens) if channel_lens else 0
        min_idx = channel_lens.index(min_len) if channel_lens else 0
        logging.info(f"[ts_code:length] - \n{ {t.ts_code:l for t, l in zip(channel_trades, channel_lens)} }"
                     f"\n -> shortest: [{channel_trades[min_idx].ts_code if channel_trades else 'N/A'} (len={min_len})]")


if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    idx_code_list = IDX_CODE_LIST
    rel_code_list = BANK_CODE_LIST_10#CODE_LIST_TEMP#ALL_CODE_LIST#BANK_CODE_LIST

    ds = StockDataset(
        ts_code=primary_stock_code,
        idx_code_list=idx_code_list,
        rel_code_list=rel_code_list,
        si=si,
        start_date='20150701',
        end_date='20251231',
        train_size=1,
        feature_type=FeatureType.BINARY_T1L10_F55,
        if_update_scaler=True,
        predict_type=PredictType.BINARY_T1L10,
        use_conv2_channel=False,
    )

    from datasets.debug_checks import (
        check_xy_alignment_conv2d,
        calc_fill_ratio_raw_channels,
        calc_fill_ratio_windowed_input,
        inspect_window_monotonic,
        inspect_window_monotonic_effective,
        check_predictable_dataset_alignment_single,
        check_predictable_dataset_shape_single,
        check_train_window_basic_single,
    )
    check_predictable_dataset_shape_single(ds, date="20201118")
    check_predictable_dataset_alignment_single(ds, date="20201118", check_t2=True)
    check_train_window_basic_single(ds, n_samples=10, use_train=True)

    # (2) 对齐检查（训练集）
    #check_xy_alignment_conv2d(ds, n_samples=10, seed=2025, use_train=True, show_channels=5)

    # (2) 对齐检查（测试集）
    # check_xy_alignment_conv2d(ds, n_samples=10, seed=2025, use_train=False, show_channels=5)

    # (3) raw 通道里 -1 占比（建议 include_y_cols=False）
    #calc_fill_ratio_raw_channels(ds, fill_val=-1.0, use_train=True, include_y_cols=False)

    # (3) 实际输入张量里 -1 占比（可选）
    #calc_fill_ratio_windowed_input(ds, fill_val=-1.0, use_train=True)

    #inspect_window_monotonic_effective(ds, use_train=True)

    #logging.info(f"ds.train_y shape: {ds.train_y.shape}, ds.test_y shape: {ds.test_y.shape}")
    #pd.set_option('display.max_columns', None)
    #start_idx = 200
    #print(f"\nraw train x sample: \n{pd.DataFrame(ds.raw_windowed_train_x[start_idx])}")
    #print(f"\nraw train y sample: \n{pd.DataFrame(ds.raw_train_y[start_idx])}")
    #print(f"feature names: {ds.get_feature_names()}")


    #raw_data, data, bp = ds.get_predictable_dataset_by_date("20151118")
    #print(f"bp: {bp}")
    #print(f"data shape: {data.shape}, data :\n{data}, \nraw_data:\n{raw_data}")
    #print(', '.join(map(str, raw_data)))
    #print(f"{ds.p_trade.remain_list}")
    #print(f"data: \n{pd.DataFrame(data[0])}")
