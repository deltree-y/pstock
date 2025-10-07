#-*- coding:UTF-8 -*-
import sys,os,logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
from itertools import chain
from stock import Stock
from stockinfo import StockInfo
from utils.utils import FeatureType, StockType, setup_logging, rolling_skew, rolling_kurtosis
from utils.tk import TOKEN

#数据说明：
#       self.raw_data_np           原始数据,包含T1,T2的数据,第一列为数据所属日期
#       self.__raw_data_pure_np    raw_data_np删除第一列后的数据
#下面数据均行数均已对齐，每个有t1t2涨跌幅的交易日为一行 - 
#       self.trade_datas           不含日期，可直接使用的所有交易数据
#       self.trade_date_list       匹配trade_datas的所有交易日期列表
#提供给dataset使用的数据:
#       self.combine_data_np       (np)包含日期(首列),所有特征,  含t1t2变化率(最后两列)
#       self.raw_data_np           (np)包含日期(首列),所有特征, 不含t1t2变化率(最后两列)

class Trade():
    def __init__(self, ts_code, si, stock_type=StockType.PRIMARY, start_date=None, end_date=None, feature_type=FeatureType.ALL):
        self.ts_code, self.si = ts_code, si
        self.start_date = start_date if start_date is not None else str(self.si.get_start_date(self.ts_code))
        self.end_date = end_date if end_date is not None else datetime.today().strftime('%Y%m%d')
        self.asset = self.si.get_asset(self.ts_code)
        self.stock_type = stock_type if self.asset == 'E' else StockType.INDEX
        self.feature_type = feature_type
        logging.debug(f"Trade::__init__() - ts_code:{ts_code}, start_date:{self.start_date}, end_date:{self.end_date}")
        self.stock = Stock(ts_code, si, self.start_date, self.end_date)
        self.trade_count = len(self.stock.df_filtered['trade_date'].values)
        logging.debug(f"[{self.stock.name}({self.ts_code})]交易数据行数:<{self.trade_count}>")
        self.y_cnt = 0  #y的列数

        #0. 获取原始数据,所有的特征都需要通过此原始数据计算得到
        self.raw_data_df = self.stock.df_filtered.copy().reset_index(drop=True)  #原始数据的DataFrame格式
        
        #1. 增删特征数据(增删对象为self.trade_df), 并返回新增特征需要丢弃的天数
        max_cut_days = self.update_new_feature()  #新增特征数据,并返回新增特征需要丢弃的天数

        #1.5 根据ts_code所属类型(pri,idx,rel),删除不需要的特征
        self.drop_features_by_type(self.stock_type)

        #2. 根据修改后的特征(self.trade_df),刷新数据
        self.raw_data_np = np.delete(self.trade_df.to_numpy(), [0], axis=1)  #删除第一列ts_code
        self.__raw_data_pure_np = np.delete(self.raw_data_np, [0], axis=1)   #删除第一列日期
        self.__trade_date_list = self.trade_df['trade_date'].values          #只保存有t1，t2数据的交易日
        
        #3. 添加预测目标值的相关数据
        self.update_t1_change_rate()
        self.update_t2_change_rate()

        #4. 统一对齐所有后续需要使用的数据(剪切头尾数据)
        self.combine_data_np, self.raw_data_np = self.get_aligned_trade_dates(max_cut_days)

    #新增特征列
    def update_new_feature(self):
        max_cut_days = 0
        #1. 创建一个交易数据的DataFrame格式,并按日期升序排列(最早日期在前),TA-lib需要按升序排列
        self.trade_df = self.raw_data_df.copy().sort_index(ascending=False).reset_index(drop=True) #按日期升序排列,方便计算

        #1.5 计算高级特征
        max_cut_days = self.extract_advanced_features()

        #2. 通过计算,新增特征数据(新增列)
        #补充日期\星期特征
        self.trade_df['date_full'] = self.trade_df['trade_date'].astype(str).astype(int)
        self.trade_df['weekday'] = pd.to_datetime(self.trade_df['trade_date'], format='%Y%m%d').dt.weekday+1

        #5个基本的技术指标
        self.trade_df['rsi_14'], max_cut_days = ta.rsi(self.trade_df['close'], length=14), max(max_cut_days, 13)
        self.trade_df['macd'], self.trade_df['macd_signal'], self.trade_df['macd_hist'] = ta.macd(self.trade_df['close'])['MACD_12_26_9'], ta.macd(self.trade_df['close'])['MACDs_12_26_9'], ta.macd(self.trade_df['close'])['MACDh_12_26_9']
        max_cut_days = max(max_cut_days, 25+8)  #macd需要25天数据才能计算出来
        self.trade_df['atr_14'], max_cut_days = ta.atr(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=14), max(max_cut_days, 14)
        self.trade_df['cci_20'], max_cut_days = ta.cci(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=20), max(max_cut_days, 19)
        self.trade_df, max_cut_days = self.trade_df.join(ta.bbands(self.trade_df['close'], length=20, std=2)), max(max_cut_days, 20)

        # 其他常用技术指标
        self.trade_df['sma_10'], max_cut_days = ta.sma(self.trade_df['close'], length=10), max(max_cut_days, 9)
        self.trade_df['ema_10'], max_cut_days = ta.ema(self.trade_df['close'], length=10), max(max_cut_days, 9)
        self.trade_df['wma_10'], max_cut_days = ta.wma(self.trade_df['close'], length=10), max(max_cut_days, 9)
        self.trade_df['stddev_10'], max_cut_days = ta.stdev(self.trade_df['close'], length=10), max(max_cut_days, 9)
        self.trade_df['roc_10'], max_cut_days = ta.roc(self.trade_df['close'], length=10), max(max_cut_days, 9)
        self.trade_df['momentum_10'], max_cut_days = ta.mom(self.trade_df['close'], length=10), max(max_cut_days, 9)
        self.trade_df, max_cut_days = self.trade_df.join(ta.adx(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=14)), max(max_cut_days, 13)
        self.trade_df['willr_14'], max_cut_days = ta.willr(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=14), max(max_cut_days, 13)
        self.trade_df['obv'], max_cut_days = ta.obv(self.trade_df['close'], self.trade_df['vol']), max(max_cut_days, 1)
        self.trade_df['cmf_20'], max_cut_days = ta.cmf(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], self.trade_df['vol'], length=20), max(max_cut_days, 19)
        self.trade_df['mfi_14'], max_cut_days = ta.mfi(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], self.trade_df['vol'], length=14), max(max_cut_days, 13)
        self.trade_df, max_cut_days = self.trade_df.join(ta.stoch(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], k=3, d=3, length=14)), max(max_cut_days, 13)
        self.trade_df['willr_14'], max_cut_days = ta.willr(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=14), max(max_cut_days, 13)
        self.trade_df['volatility_20'], max_cut_days = ta.volatility(self.trade_df['close'], length=20, std=2), max(max_cut_days, 19)
        self.trade_df['natr_14'], max_cut_days = ta.natr(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=14), max(max_cut_days, 13)
        max_cut_days =max(max_cut_days, 27)

        # 近N日成交量均值/中位数/最大值
        for win in [5, 10, 20]:
            self.trade_df[f'vol_mean_{win}d'] = self.trade_df['vol'].rolling(win).mean()
            self.trade_df[f'vol_ratio_{win}d'] = self.trade_df['vol'] / self.trade_df[f'vol_mean_{win}d']
            self.trade_df[f'vol_max_{win}d'] = self.trade_df['vol'].rolling(win).max()
            self.trade_df[f'is_vol_break_{win}d'] = (self.trade_df['vol'] > self.trade_df[f'vol_max_{win}d'].shift(1)).astype(int)
        max_cut_days =max(max_cut_days, 20)

        # 尾盘拉升：收盘价与最低价的距离/全幅度
        self.trade_df['close_vs_low'] = (self.trade_df['close'] - self.trade_df['low']) / (self.trade_df['high'] - self.trade_df['low'] + 1e-6)
        # 或者：收盘价与开盘价比
        self.trade_df['close_vs_open'] = (self.trade_df['close'] - self.trade_df['open']) / self.trade_df['open']

        self.trade_df.fillna(0,inplace=True)

        #3. 将trade_df按日期降序排列(最新日期在前),方便后续使用
        self.trade_df = self.trade_df.copy().sort_index(ascending=False).reset_index(drop=True) 
        return max_cut_days

    def extract_advanced_features(self):
        """提取高级特征"""
        # 假设raw_data是包含价格和交易量的DataFrame
        #df = self.trade_df
        
        # 1. 提取价格动量特征
        self.trade_df['return_1d'] = self.trade_df['close'].pct_change(1)
        self.trade_df['return_5d'] = self.trade_df['close'].pct_change(5)
        self.trade_df['return_10d'] = self.trade_df['close'].pct_change(10)
        
        # 2. 波动率特征
        self.trade_df['volatility_5d'] = self.trade_df['return_1d'].rolling(5).std()
        self.trade_df['volatility_10d'] = self.trade_df['return_1d'].rolling(10).std()
        
        # 3. 价格与成交量关系
        self.trade_df['price_volume_ratio'] = self.trade_df['close'] / self.trade_df['vol']
        self.trade_df['volume_change'] = self.trade_df['vol'].pct_change(1)
        
        # 4. 高级统计特征
        #self.trade_df['return_skew_5d'] = self.trade_df['return_1d'].rolling(5).apply(lambda x: skew(x))
        #self.trade_df['return_kurt_5d'] = self.trade_df['return_1d'].rolling(5).apply(lambda x: kurtosis(x))
        arr = self.trade_df['return_1d'].to_numpy()
        self.trade_df['return_skew_5d'] = rolling_skew(arr, 5)
        self.trade_df['return_kurt_5d'] = rolling_kurtosis(arr, 5)
        # 5. 非线性变换
        self.trade_df['log_return'] = np.log(self.trade_df['close'] / self.trade_df['close'].shift(1))
        self.trade_df['log_volume'] = np.log(self.trade_df['vol'])

        #### 第二批新增技术指标 ####
        # 1. 前20日高点
        self.trade_df['high_20d_max'] = self.trade_df['high'].rolling(20).max()
        self.trade_df['close_to_high_20d'] = (self.trade_df['close'] - self.trade_df['high_20d_max']) / self.trade_df['high_20d_max']

        # 2. 创新高标记
        self.trade_df['is_new_high_20d'] = (self.trade_df['high'] == self.trade_df['high_20d_max']).astype(int)

        # 3. 振幅
        self.trade_df['amplitude'] = (self.trade_df['high'] - self.trade_df['low']) / ((self.trade_df['open'] + self.trade_df['close']) / 2)

        # 4. 成交量放大倍数
        self.trade_df['vol_ratio_20d'] = self.trade_df['vol'] / self.trade_df['vol'].rolling(20).mean()

        # 5. MACD金叉死叉（假设已用ta库）
        macd = ta.macd(self.trade_df['close'])
        self.trade_df['macd_cross'] = ((macd['MACD_12_26_9'] > macd['MACDs_12_26_9']).astype(int))
        
        max_cut_days = 26  # 新增特征需要丢弃的天数为26天

        return max_cut_days


    #根据数据类型,删除不需要的特征
    def drop_features_by_type(self, stock_type):
        if stock_type == StockType.PRIMARY or stock_type == StockType.RELATED:
            #皮尔逊+互信息+树模型交集特征
            basic_features = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'industry_idx', 'date_full']#, 'stock_idx']
            t1l10_features_25 = ['natr_14', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'dv_ratio', '1y', 'turnover_rate_f', '6m', 'cmt', 'y20', 'ltc', 'y1', 'pb', 'y30', 'w52_ce', 'w26_bd', 'y30_us_trycr', 'y10', 'y10_us_trycr', 'w26_ce']
            t1l10_features_35 = ['natr_14', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'dv_ratio', '1y', 'turnover_rate_f', '6m', 'cmt', 'y20', 'ltc', 'y1', 'pb', 'y30', 'w52_ce', 'w26_bd', 'y30_us_trycr', 'y10', 'y10_us_trycr', 'w26_ce', 'ltr_avg', 'w52_bd', 'y5_us_trycr', 'y5', '1w', 'on', 'm1', 'w4_bd', 'w4_ce', 'pe', 'total_mv', 'atr_14', 'stddev_10']
            t1l10_features_45 = ['natr_14', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'dv_ratio', '1y', 'turnover_rate_f', '6m', 'cmt', 'y20', 'ltc', 'y1', 'pb', 'y30', 'w52_ce', 'w26_bd', 'y30_us_trycr', 'y10', 'y10_us_trycr', 'w26_ce', 'ltr_avg', 'w52_bd', 'y5_us_trycr', 'y5', '1w', 'on', 'm1', 'w4_bd', 'w4_ce', 'pe', 'total_mv', 'atr_14', 'stddev_10', 'ps', 'ADX_14', 'log_volume', 'DMP_14', 'amount', 'return_10d', 'roc_10', 'BBU_20_2.0', 'return_5d']
            t1l10_features_55 = ['natr_14', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'dv_ratio', '1y', 'turnover_rate_f', '6m', 'cmt', 'y20', 'ltc', 'y1', 'pb', 'y30', 'w52_ce', 'w26_bd', 'y30_us_trycr', 'y10', 'y10_us_trycr', 'w26_ce', 'ltr_avg', 'w52_bd', 'y5_us_trycr', 'y5', '1w', 'on', 'm1', 'w4_bd', 'w4_ce', 'pe', 'total_mv', 'atr_14', 'stddev_10', 'ps', 'ADX_14', 'log_volume', 'DMP_14', 'amount', 'return_10d', 'roc_10', 'BBU_20_2.0', 'return_5d', 'e_factor', 'sma_10', 'ema_10', 'wma_10', 'BBM_20_2.0', 'pre_close', 'obv']
            
            t1l05_features_35 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1']
            t1l05_features_55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']

            t1l15_features_35 = ['natr_14', 'amplitude', 'volatility_10d', 'BBB_20_2.0', 'e_factor', 'dv_ratio', 'volatility_5d', 'turnover_rate_f', 'date_full', 'close_to_high_20d', 'pb', 'macd_cross', 'pe', 'y5_us_trycr', 'total_mv', 'y10_us_trycr', 'ltr_avg', 'y1', 'w52_bd', 'y30_us_trycr', 'w52_ce', 'w26_bd', 'w26_ce', 'w4_bd', 'm1', 'w4_ce', '1y', 'y5', 'y10', 'cmt', 'y20', 'ltc', 'y30', '6m', ]
            t1l15_features_55 = ['natr_14', 'amplitude', 'volatility_10d', 'BBB_20_2.0', 'e_factor', 'dv_ratio', 'volatility_5d', 'turnover_rate_f', 'date_full', 'close_to_high_20d', 'pb', 'macd_cross', 'pe', 'y5_us_trycr', 'total_mv', 'y10_us_trycr', 'ltr_avg', 'y1', 'w52_bd', 'y30_us_trycr', 'w52_ce', 'w26_bd', 'w26_ce', 'w4_bd', 'm1', 'w4_ce', '1y', 'y5', 'y10', 'cmt', 'y20', 'ltc', 'y30', '6m', 'vol_max_20d', '1w', 'atr_14', 'vol_max_10d', 'on', 'ADX_14', 'stddev_10', 'vol_max_5d', 'DMP_14', 'ps', 'return_10d', 'roc_10', 'industry_idx', 'rsi_14', 'amount', 'return_5d', 'log_volume', 'high_20d_max', 'weekday', 'cmf_20']
            t1l15_features_75 = ['natr_14', 'amplitude', 'volatility_10d', 'BBB_20_2.0', 'e_factor', 'dv_ratio', 'volatility_5d', 'turnover_rate_f', 'date_full', 'close_to_high_20d', 'pb', 'macd_cross', 'pe', 'y5_us_trycr', 'total_mv', 'y10_us_trycr', 'ltr_avg', 'y1', 'w52_bd', 'y30_us_trycr', 'w52_ce', 'w26_bd', 'w26_ce', 'w4_bd', 'm1', 'w4_ce', '1y', 'y5', 'y10', 'cmt', 'y20', 'ltc', 'y30', '6m', 'vol_max_20d', '1w', 'atr_14', 'vol_max_10d', 'on', 'ADX_14', 'stddev_10', 'vol_max_5d', 'DMP_14', 'ps', 'return_10d', 'roc_10', 'industry_idx', 'rsi_14', 'amount', 'return_5d', 'log_volume', 'high_20d_max', 'weekday', 'cmf_20', 'obv', 'vol_ratio_20d', 'willr_14', 'BBU_20_2.0', 'macd_signal', 'mfi_14', 'macd', 'close_vs_open', 'sma_10', 'ema_10', 'BBM_20_2.0', 'close', 'pre_close', 'high', 'open', 'wma_10', 'cci_20', 'low', 'BBP_20_2.0', 'close_vs_low']

            t2h10_features_25 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10']
            t2h10_features_35 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce']
            t2h10_features_45 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10']
            t2h10_features_55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            advanced_features = ['return_1d', 'volatility_5d', 'is_new_high_20d', 'is_vol_break_5d', 'close_vs_low', 'close_vs_open']
            remain_list = list(dict.fromkeys(chain(basic_features, locals()[self.feature_type.value], advanced_features))) if self.feature_type != FeatureType.ALL else self.trade_df.columns.to_list()
            #logging.info(f"After feature selection, remain {len(remain_list)}")
            self.col_low, self.col_high, self.col_close = remain_list.index('low')-2, remain_list.index('high')-2, remain_list.index('close')-2 #在raw_data_np中的列索引位置,需要-2(减去ts_code和trade_date两列)
        elif stock_type == StockType.RELATED:
            pass
        elif stock_type == StockType.INDEX:
            remain_list = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
        else:
            logging.error(f"Unknown stock type:{stock_type}, no features dropped!")
            return
        self.remain_list = remain_list
        self.trade_df = self.trade_df[remain_list]
        logging.debug(f"After drop features by type({stock_type}), trade_df columns are:\n{self.trade_df.columns.to_list()}")

    #对齐所有后续需要使用的数据
    #max_cut_days表示新增特征需要丢弃的天数
    def get_aligned_trade_dates(self, max_cut_days):
        if self.trade_count < 2:
            logging.error("Not enough data to align trade dates.")
            return [],[],[],[]
        #1. 根据数据剪掉部分头部数据,进行对齐
        self.__trade_datas = self.__raw_data_pure_np[2:, :]     #从第三天开始对齐
        self.__trade_date_list = self.__trade_date_list[2:]  #从第三天开始对齐
        self.__t1l_change_rate = self.__t1l_change_rate[1:]    #从第二天开始对齐
        self.__t1h_change_rate = self.__t1h_change_rate[1:]    #从第二天开始对齐
        #self.t2l_change_rate = self.t2l_change_rate    #从第三天开始对齐,不需要变动
        #self.t2h_change_rate = self.t2h_change_rate    #从第三天开始对齐,不需要变动

        #2. 组合生成头部对齐的数据
        #self.combine_data_np = np.column_stack((self.__trade_date_list, self.__trade_datas, self.__t1l_change_rate, self.__t2h_change_rate))
        self.combine_data_np = np.column_stack((self.__trade_date_list, self.__trade_datas, self.__t1l_change_rate, self.__t1h_change_rate, self.__t2l_change_rate, self.__t2h_change_rate))
        self.y_cnt = 4  #y的列数,表示t1l,t1h,t2l,t2h变化率
        
        #3. 统一剪掉尾部并返回
        return self.combine_data_np[:-max_cut_days], self.raw_data_np[:-max_cut_days]

    #计算t1,t2变化率
    #t1_change_rate表示T1低值的变化率
    def update_t1_change_rate(self):#表示T1低值的变化率
        ##计算说明:
        # t1l_change_rate = t1_low  - t0_close / t0_close
        # t1h_change_rate = t1_high - t0_close / t0_close
        if self.trade_count < 2:
            logging.error("Not enough data to calculate T1 change rate.")
            return
        self.__t1l_change_rate = (self.__raw_data_pure_np[:-1, self.col_low] - self.__raw_data_pure_np[1:, self.col_close]) / self.__raw_data_pure_np[1:, self.col_close] \
            if self.stock_type != StockType.INDEX else self.__raw_data_pure_np[:-1, 0]-self.__raw_data_pure_np[:-1, 0]
        self.__t1h_change_rate = (self.__raw_data_pure_np[:-1, self.col_high] - self.__raw_data_pure_np[1:, self.col_close]) / self.__raw_data_pure_np[1:, self.col_close] \
            if self.stock_type != StockType.INDEX else self.__raw_data_pure_np[:-1, 0]-self.__raw_data_pure_np[:-1, 0]
        #self.t1_change_rate = np.array([RateCat(rate=x,scale=T1L_SCALE).get_label() for x in self.t1_change_rate])
    
    #t2_change_rate表示T2高值的变化率
    def update_t2_change_rate(self):#表示T2高值的变化率
        ##计算说明:
        # t2l_change_rate = t2_low  - t0_close / t0_close
        # t2h_change_rate = t2_high - t0_close / t0_close
        if self.trade_count < 3:
            logging.error("Not enough data to calculate T2 change rate.")
            return
        self.__t2l_change_rate = (self.__raw_data_pure_np[:-2, self.col_low] - self.__raw_data_pure_np[2:, self.col_close]) / self.__raw_data_pure_np[2:, self.col_close] \
            if self.stock_type != StockType.INDEX else self.__raw_data_pure_np[:-2, 0]-self.__raw_data_pure_np[:-2, 0]
        self.__t2h_change_rate = (self.__raw_data_pure_np[:-2, self.col_high] - self.__raw_data_pure_np[2:, self.col_close]) / self.__raw_data_pure_np[2:, self.col_close] \
            if self.stock_type != StockType.INDEX else self.__raw_data_pure_np[:-2, 0]-self.__raw_data_pure_np[:-2, 0]
        #self.t2_change_rate = np.array([RateCat(rate=x,scale=T2H_SCALE).get_label() for x in self.t2_change_rate])


if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    #ts_code = '000001.SH'
    ts_code = '600036.SH'
    t = Trade(ts_code, si, stock_type=StockType.RELATED, start_date='20250101', end_date='20250829')
    #t = Trade(ts_code, si, stock_type=StockType.INDEX, start_date='20250101', end_date='20250829')
    print(f"trade shape: {t.combine_data_np.shape}, raw shape: {t.raw_data_np.shape}")
    print(f"trade head:\n{pd.DataFrame(t.raw_data_np).head(5)}\ntrade tail:\n{pd.DataFrame(t.raw_data_np).tail(5)}")