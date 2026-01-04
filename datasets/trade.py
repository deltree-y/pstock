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
from utils.const_def import MIN_TRADE_DATA_ROWS

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
    def __init__(self, ts_code, si:StockInfo, stock_type=StockType.PRIMARY, start_date=None, end_date=None, feature_type=FeatureType.ALL):
        self.ts_code, self.si = ts_code, si
        self.asset = self.si.get_asset(self.ts_code)
        self.start_date = start_date if start_date is not None else str(self.si.get_start_date(self.ts_code, self.asset))
        self.end_date = end_date if end_date is not None else datetime.today().strftime('%Y%m%d')
        self.stock_type = stock_type if self.asset == 'E' else StockType.INDEX
        self.feature_type = feature_type
        logging.debug(f"Trade::__init__() - ts_code:{ts_code}, start_date:{self.start_date}, end_date:{self.end_date}")
        self.stock = Stock(ts_code, si, self.start_date, self.end_date)
        self.trade_count = len(self.stock.df_filtered['trade_date'].values)
        if self.trade_count < MIN_TRADE_DATA_ROWS:
            logging.error(f"[{self.stock.name}({self.ts_code})]交易数据行数:<{self.trade_count}>, 无法进行后续处理!")
            return None
        logging.debug(f"[{self.stock.name}({self.ts_code})]交易数据行数:<{self.trade_count}>")
        self.y_cnt = 0  #y的列数

        #0. 获取原始数据,所有的特征都需要通过此原始数据计算得到
        self.raw_data_df = self.stock.df_filtered.copy().reset_index(drop=True)  #原始数据的DataFrame格式
        dates = self.raw_data_df['trade_date'].astype(int).to_numpy()
        if dates.shape[0] >= 2 and not np.all(dates[:-1] <= dates[1:]):
            logging.warning("Trade raw_data_df trade_date is not ascending (old->new).")
        
        #1. 增删特征数据(增删对象为self.trade_df), 并返回新增特征需要丢弃的天数
        self.max_cut_days = self.update_new_feature()  #新增特征数据,并返回新增特征需要丢弃的天数

        #1.5 根据ts_code所属类型(pri,idx,rel),删除不需要的特征
        self.drop_features_by_type(self.stock_type)

        #2. 根据修改后的特征(self.trade_df),刷新数据
        self.raw_data_np = np.delete(self.trade_df.to_numpy(), [0], axis=1)  #删除第一列ts_code, 保留日期列
        self.__raw_data_pure_np = np.delete(self.raw_data_np, [0], axis=1)   #删除第一列日期, 只保留特征数据
        self.__trade_date_list = self.trade_df['trade_date'].values          #只保存有t1，t2数据的交易日
        #print(f"DEBUG:__trade_date_list len:{len(self.__trade_date_list)}, head:{self.__trade_date_list[:3]}, tail:{self.__trade_date_list[-3:]}")
        
        #3. 添加预测目标值的相关数据
        self.update_t1_change_rate()
        self.update_t2_change_rate()

        #TODO: 此处要仔细分析,是否要修改
        #4. 统一对齐所有后续需要使用的数据(剪切头尾数据)
        self.combine_data_np, self.raw_data_np = self.get_aligned_trade_dates(self.max_cut_days)

    #新增特征列
    def update_new_feature(self):
        max_cut_days = 0
        #1. 创建一个交易数据的DataFrame格式的拷贝数据, TA-lib需要按升序排列
        self.trade_df = self.raw_data_df.copy().reset_index(drop=True) #此处日期已为升序排列(旧->新)
        #print(f"先日期升序: self.trade_df head \n{self.trade_df.head(3)}")

        #1.5 计算高级特征
        max_cut_days = self.extract_advanced_features()

        #2. 通过计算,新增特征数据(新增列)
        #补充日期\星期特征
        self.trade_df['date_full'] = self.trade_df['trade_date'].astype(str).astype(int)
        self.trade_df['weekday'] = pd.to_datetime(self.trade_df['trade_date'], format='%Y%m%d').dt.weekday+1

        #2.1 在增删特征前，确保索引为DatetimeIndex(新增[20251016])
        if not isinstance(self.trade_df.index, pd.DatetimeIndex):
            self.trade_df['trade_date'] = pd.to_datetime(self.trade_df['trade_date'], format='%Y%m%d')
            self.trade_df.set_index('trade_date', inplace=True)
        #print(f"after update_new_feature 2.1: self.trade_df head \n{self.trade_df.head(3)}")

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
        # VWAP
        self.trade_df['vwap'] = ta.vwap(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], self.trade_df['vol'])
        # SuperTrend（需pandas_ta 0.3.14+）
        st = ta.supertrend(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'])
        self.trade_df['supertrend'] = st['SUPERT_7_3.0']
        # Donchian Channel
        dc = ta.donchian(self.trade_df['high'], self.trade_df['low'], lower_length=20, upper_length=20)
        self.trade_df['donchian_upper'] = dc['DCU_20_20']
        self.trade_df['donchian_lower'] = dc['DCL_20_20']
        self.trade_df['donchian_mid']   = dc['DCM_20_20']

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
        
        self.trade_df.fillna(0,inplace=True)    #填充新增特征的NaN值为0

        #2.99 所有特征添加完后，重置索引恢复trade_date为列(新增[20251016])
        self.trade_df.insert(1, 'trade_date', self.trade_df.index.strftime('%Y%m%d').astype(str))
        self.trade_df.reset_index(drop=True, inplace=True)
        #print(f"after update_new_feature 2.99 self.trade_df head \n{self.trade_df.head(3)}, \ntail \n{self.trade_df.tail(3)}")


        #TODO: 此处要修改
        #3. 将trade_df按日期降序排列(最新日期在前),方便后续使用
        #self.trade_df = self.trade_df.copy().sort_index(ascending=False).reset_index(drop=True) 
        #print(f"再日期降序: self.trade_df head \n{self.trade_df.head(3)}")
        #print(f"after update_new_feature 3 self.trade_df head \n{self.trade_df.head(3)}")

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
        arr = self.trade_df['return_1d'].to_numpy()
        self.trade_df['return_skew_5d'] = rolling_skew(arr, 5)
        self.trade_df['return_kurt_5d'] = rolling_kurtosis(arr, 5)

        # 5. 非线性变换
        self.trade_df['log_return'] = np.log(self.trade_df['close'] / self.trade_df['close'].shift(1))
        self.trade_df['log_volume'] = np.log(self.trade_df['vol'])

        #### 第二批新增技术指标 ####

        # 0. 史高/史低相关特征
        self.trade_df['his_high_all'] = self.trade_df['high'].cummax()
        self.trade_df['his_low_all'] = self.trade_df['low'].cummin()
        self.trade_df['close_to_his_high'] = (self.trade_df['close'] - self.trade_df['his_high_all']) / self.trade_df['his_high_all']
        self.trade_df['close_to_his_low'] = (self.trade_df['close'] - self.trade_df['his_low_all']) / self.trade_df['his_low_all']
        self.trade_df['is_new_his_high'] = (self.trade_df['high'] == self.trade_df['his_high_all']).astype(int)

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
            basic_features = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close']#, 'stock_idx']
            
            regress_t1l_f55 = ['date_full', 'dv_ratio', 'high', 'vol', 'y5_us_trycr', '6m', 'on', 'turnover_rate_f', 'vol_max_10d', 'w52_bd', 'close_to_his_high', 'close_vs_low', 'obv', 'atr_14', 'm1', 'total_mv', 'vol_mean_10d', 'y20', 'y30', 'w4_ce', 'industry_idx', 'volatility_5d', 'sell_elg_vol', 'BBB_20_2.0', 'close_to_high_20d', 'low', 'ltc', 'return_5d', 'close_to_his_low', 'vol_mean_20d', 'w4_bd', 'ltr_avg', 'w26_ce', 'close_vs_open', 'w52_ce', 'vol_max_5d', 'w26_bd', 'macd_signal', 'y10', 'y5', 'y10_us_trycr', 'vol_max_20d', 'volatility_10d', 'y30_us_trycr', 'natr_14', 'stddev_10', 'return_10d', 'open', 'pe', 'ps', 'y1', 'close', 'amplitude', '1y', 'roc_10', '1w', 'pb', 'buy_elg_vol', 'cmt']
            regress_t1l_f50 = regress_t1l_f55[:50]
            regress_t1h_f55 = ['date_full', 'dv_ratio', 'high', 'vol', 'y5_us_trycr', '6m', 'on', 'turnover_rate_f', 'vol_max_10d', 'w52_bd', 'close_to_his_high', 'close_vs_low', 'obv', 'atr_14', 'm1', 'total_mv', 'vol_mean_10d', 'y20', 'y30', 'w4_ce', 'industry_idx', 'volatility_5d', 'sell_elg_vol', 'BBB_20_2.0', 'close_to_high_20d', 'low', 'ltc', 'return_5d', 'close_to_his_low', 'vol_mean_20d', 'w4_bd', 'ltr_avg', 'w26_ce', 'close_vs_open', 'w52_ce', 'vol_max_5d', 'w26_bd', 'macd_signal', 'y10', 'y5', 'y10_us_trycr', 'vol_max_20d', 'volatility_10d', 'y30_us_trycr', 'natr_14', 'stddev_10', 'return_10d', 'open', 'pe', 'ps', 'y1', 'close', 'amplitude', '1y', 'roc_10', '1w', 'pb', 'buy_elg_vol', 'cmt']
            regress_t1h_f50 = regress_t1h_f55[:50]
            regress_t2h_f55 = ['date_full', 'dv_ratio', 'high', 'vol', 'y5_us_trycr', '6m', 'on', 'turnover_rate_f', 'vol_max_10d', 'w52_bd', 'close_to_his_high', 'close_vs_low', 'obv', 'atr_14', 'm1', 'total_mv', 'vol_mean_10d', 'y20', 'y30', 'w4_ce', 'industry_idx', 'volatility_5d', 'sell_elg_vol', 'BBB_20_2.0', 'close_to_high_20d', 'low', 'ltc', 'return_5d', 'close_to_his_low', 'vol_mean_20d', 'w4_bd', 'ltr_avg', 'w26_ce', 'close_vs_open', 'w52_ce', 'vol_max_5d', 'w26_bd', 'macd_signal', 'y10', 'y5', 'y10_us_trycr', 'vol_max_20d', 'volatility_10d', 'y30_us_trycr', 'natr_14', 'stddev_10', 'return_10d', 'open', 'pe', 'ps', 'y1', 'close', 'amplitude', '1y', 'roc_10', '1w', 'pb', 'buy_elg_vol', 'cmt']
            regress_t2h_f50 = regress_t2h_f55[:50]
            regress_t2h_f30 = regress_t2h_f55[:30]
            regress_t2h_f20 = regress_t2h_f55[:20]

            binary_t1l03_f55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']
            binary_t1l04_f55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']
            binary_t1l05_f55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']
            binary_t1l05_f50 = binary_t1l05_f55[:50]
            binary_t1l05_f35 = binary_t1l05_f55[:35]
            binary_t1l06_f55 = ['y5', 'close_vs_low', 'cmt', 'w52_bd', 'w26_bd', 'return_5d', 'date_full', 'ltc', 'ltr_avg', 'vol', 'willr_14', 'amplitude', 'ps', 'atr_14', 'm1', 'w4_ce', 'turnover_rate_f', 'w52_ce', 'y20', 'close', 'macd_hist', 'DMP_14', 'y5_us_trycr', '1y', 'w26_ce', 'open', 'stddev_10', 'w4_bd', 'volatility_10d', 'pe', 'pb', 'close_to_high_20d', 'dv_ratio', 'volatility_5d', 'y10', 'y10_us_trycr', 'obv', '1w', 'BBB_20_2.0', 'on', 'natr_14', 'macd_signal', 'y1', 'ADX_14', '6m', 'total_mv', 'y30_us_trycr', 'y30']
            binary_t1l07_f55 = ['y5', 'close_vs_low', 'cmt', 'w52_bd', 'w26_bd', 'return_5d', 'date_full', 'ltc', 'ltr_avg', 'vol', 'willr_14', 'amplitude', 'ps', 'atr_14', 'm1', 'w4_ce', 'turnover_rate_f', 'w52_ce', 'y20', 'close', 'macd_hist', 'DMP_14', 'y5_us_trycr', '1y', 'w26_ce', 'open', 'stddev_10', 'w4_bd', 'volatility_10d', 'pe', 'pb', 'close_to_high_20d', 'dv_ratio', 'volatility_5d', 'y10', 'y10_us_trycr', 'obv', '1w', 'BBB_20_2.0', 'on', 'natr_14', 'macd_signal', 'y1', 'ADX_14', '6m', 'total_mv', 'y30_us_trycr', 'y30']
            binary_t1l08_f55 = ['y5', 'close_vs_low', 'cmt', 'w52_bd', 'w26_bd', 'return_5d', 'date_full', 'ltc', 'ltr_avg', 'vol', 'willr_14', 'amplitude', 'ps', 'atr_14', 'm1', 'w4_ce', 'turnover_rate_f', 'w52_ce', 'y20', 'close', 'macd_hist', 'DMP_14', 'y5_us_trycr', '1y', 'w26_ce', 'open', 'stddev_10', 'w4_bd', 'volatility_10d', 'pe', 'pb', 'close_to_high_20d', 'dv_ratio', 'volatility_5d', 'y10', 'y10_us_trycr', 'obv', '1w', 'BBB_20_2.0', 'on', 'natr_14', 'macd_signal', 'y1', 'ADX_14', '6m', 'total_mv', 'y30_us_trycr', 'y30']
            binary_t1l08_f30 = ['atr_14', 'vol_max_20d', 'y5_us_trycr', 'close_to_high_20d', 'close', 'volatility_5d', 'volatility_10d', 'dv_ratio', 'ltr_avg', 'pe', 'obv', 'y30_us_trycr', 'vol', 'amplitude', 'natr_14', 'y10_us_trycr', 'low', 'stddev_10', '1w', 'close_vs_low', 'pb', 'open', 'high', 'BBB_20_2.0', 'total_mv', 'on', 'date_full', 'turnover_rate_f', '6m', 'y30']

            binary_t1l10_f55 = ['y5', 'close_vs_low', 'cmt', 'w52_bd', 'w26_bd', 'return_5d', 'date_full', 'ltc', 'ltr_avg', 'vol', 'willr_14', 'amplitude', 'ps', 'atr_14', 'm1', 'w4_ce', 'turnover_rate_f', 'w52_ce', 'y20', 'close', 'macd_hist', 'DMP_14', 'y5_us_trycr', '1y', 'w26_ce', 'open', 'stddev_10', 'w4_bd', 'volatility_10d', 'pe', 'pb', 'close_to_high_20d', 'dv_ratio', 'volatility_5d', 'y10', 'y10_us_trycr', 'obv', '1w', 'BBB_20_2.0', 'on', 'natr_14', 'macd_signal', 'y1', 'ADX_14', '6m', 'total_mv', 'y30_us_trycr', 'y30']
            binary_t1l10_f50 = binary_t1l10_f55[:50]
            binary_t1l10_f35 = binary_t1l10_f55[:35]
            binary_t1l10_f15 = binary_t1l10_f55[:15]

            binary_t1l15_f75 = ['natr_14', 'amplitude', 'volatility_10d', 'BBB_20_2.0', 'e_factor', 'dv_ratio', 'volatility_5d', 'turnover_rate_f', 'date_full', 'close_to_high_20d', 'pb', 'macd_cross', 'pe', 'y5_us_trycr', 'total_mv', 'y10_us_trycr', 'ltr_avg', 'y1', 'w52_bd', 'y30_us_trycr', 'w52_ce', 'w26_bd', 'w26_ce', 'w4_bd', 'm1', 'w4_ce', '1y', 'y5', 'y10', 'cmt', 'y20', 'ltc', 'y30', '6m', 'vol_max_20d', '1w', 'atr_14', 'vol_max_10d', 'on', 'ADX_14', 'stddev_10', 'vol_max_5d', 'DMP_14', 'ps', 'return_10d', 'roc_10', 'industry_idx', 'rsi_14', 'amount', 'return_5d', 'log_volume', 'high_20d_max', 'weekday', 'cmf_20', 'obv', 'vol_ratio_20d', 'willr_14', 'BBU_20_2.0', 'macd_signal', 'mfi_14', 'macd', 'close_vs_open', 'sma_10', 'ema_10', 'BBM_20_2.0', 'close', 'pre_close', 'high', 'open', 'wma_10', 'cci_20', 'low', 'BBP_20_2.0', 'close_vs_low']
            binary_t1l15_f55 = binary_t1l15_f75[:55]
            binary_t1l15_f50 = binary_t1l15_f75[:50]
            binary_t1l15_f35 = binary_t1l15_f75[:35]

            binary_t1h05_f55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']
            binary_t1h05_f50 = binary_t1h05_f55[:50]
            binary_t1h05_f35 = binary_t1h05_f55[:35]
            binary_t1h06_f55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']
            binary_t1h07_f55 = ['natr_14', 'amplitude', 'dv_ratio', 'e_factor', 'volatility_10d', 'BBB_20_2.0', 'volatility_5d', 'date_full', 'total_mv', 'turnover_rate_f', 'on', 'pb', 'pe', 'close_to_high_20d', 'macd_cross', '1y', '6m', '1w', 'log_volume', 'vol_max_20d', 'vol_max_10d', 'atr_14', 'y5_us_trycr', 'ps', 'ADX_14', 'stddev_10', 'weekday', 'w26_bd', 'vol_max_5d', 'y10_us_trycr', 'w52_ce', 'w52_bd', 'w26_ce', 'y1', 'DMP_14', 'industry_idx', 'm1', 'w4_bd', 'w4_ce', 'ltr_avg', 'y5', 'y30_us_trycr', 'y10', 'vol_mean_20d', 'y20', 'ltc', 'roc_10', 'rsi_14', 'return_10d', 'cmt', 'buy_elg_vol', 'vol_ratio_20d', 'sell_elg_vol', 'y30']
            binary_t1h08_f18 = ['pe', 'turnover_rate_f', 'on', 'volatility_10d', 'volatility_5d', '1w', 'pb', 'BBB_20_2.0', 'atr_14', 'vol_max_10d', 'dv_ratio', 'vol', 'total_mv', 'vol_max_20d', 'amplitude', 'DMP_14', 'close_to_high_20d', 'natr_14']

            binary_t1h10_f75 = ['natr_14', 'e_factor', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', 'volatility_5d', 'macd_cross', 'date_full', 'total_mv', 'turnover_rate_f', 'pb', 'pe', 'close_to_high_20d', 'vol_max_20d', 'vol_max_10d', 'w52_bd', 'w52_ce', 'y1', 'vol_max_5d', 'cmt', 'y10', 'y5', 'ltc', 'w26_ce', 'y20', 'y30', 'w26_bd', 'm1', 'w4_bd', '1w', 'y5_us_trycr', 'w4_ce', 'y10_us_trycr', 'on', '1y', 'y30_us_trycr', 'ltr_avg', 'log_volume', 'atr_14', '6m', 'stddev_10', 'DMP_14', 'rsi_14', 'ps', 'ADX_14', 'willr_14', 'roc_10', 'return_10d', 'weekday', 'obv', 'return_5d', 'sell_elg_vol', 'vol_mean_20d', 'BBP_20_2.0', 'vol_mean_10d', 'cci_20', 'buy_elg_vol', 'industry_idx', 'vol_mean_5d', 'DMN_14', 'close_vs_open', 'mfi_14', 'pct_chg', 'return_1d', 'high_20d_max', 'vol', 'sell_sm_vol', 'log_return', 'buy_sm_vol', 'BBU_20_2.0', 'STOCHk_3_3_3', 'buy_md_vol', 'sell_lg_vol', 'buy_lg_vol']
            binary_t1h10_f55 = binary_t1h10_f75[:55]
            binary_t1h10_f50 = binary_t1h10_f75[:50]
            binary_t1h10_f35 = binary_t1h10_f75[:35]

            binary_t2h03_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            binary_t2h04_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            binary_t2h05_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            binary_t2h06_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            binary_t2h07_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            binary_t2h08_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']

            binary_t2h10_f55 = ['e_factor', 'natr_14', 'date_full', 'macd_cross', 'vol_max_5d', 'volatility_10d', 'amplitude', 'dv_ratio', 'BBB_20_2.0', '1y', 'on', 'vol_max_10d', '6m', 'volatility_5d', '1w', 'vol_max_20d', 'total_mv', 'close_to_high_20d', 'pe', 'w52_bd', 'turnover_rate_f', 'y1', 'pb', 'w52_ce', 'y10', 'y5', 'w26_bd', 'w26_ce', 'y5_us_trycr', 'ltc', 'y20', 'cmt', 'y30', 'w4_bd', 'w4_ce', 'm1', 'y10_us_trycr', 'log_volume', 'y30_us_trycr', 'ltr_avg', 'atr_14', 'rsi_14', 'weekday', 'stddev_10', 'DMP_14', 'ps', 'willr_14', 'ADX_14', 'STOCHk_3_3_3', 'roc_10', 'return_10d', 'return_5d', 'sell_elg_vol', 'buy_elg_vol']
            binary_t2h10_f50 = binary_t2h10_f55[:50]
            binary_t2h10_f45 = binary_t2h10_f55[:45]
            binary_t2h10_f35 = binary_t2h10_f55[:35]
            binary_t2h10_f25 = binary_t2h10_f55[:25]
            binary_t2h10_f10 = []

            classify_features_50 = ['low', 'm1', 'y10', 'y20', 'close', 'amplitude', 'cmt', 'ADX_14', 'atr_14', 'dv_ratio', 'open', 'pb', 'w52_bd', 'w52_ce', 'natr_14', 'date_full', 'ltc', 'volatility_5d', 'return_5d', 'BBB_20_2.0', 'on', 'ltr_avg', 'y1', '1y', 'y30_us_trycr', 'vol', 'macd_signal', 'y30', 'y10_us_trycr', 'w4_ce', '1w', 'obv', 'volatility_10d', 'w4_bd', '6m', 'y5', 'pe', 'DMP_14', 'high', 'turnover_rate_f', 'stddev_10', 'w26_bd', 'macd_hist', 'total_mv', 'w26_ce', 'ps', 'close_vs_low', 'y5_us_trycr', 'close_to_high_20d', 'net_mf_vol']
            classify_features_30 = classify_features_50[:30]

            #advanced_features = ['industry_idx', 'date_full', 'vwap', 'supertrend', 'donchian_upper', 'donchian_lower', 'donchian_mid', 'high_20d_max', 'close_to_high_20d', 'is_new_high_20d', 'his_high_all', 'close_to_his_high', 'his_low_all', 'close_to_his_low', 'is_new_his_high']#, 'his_low', 'his_high', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'weight_avg', 'winner_rate']
            advanced_features = ['industry_idx', 'date_full', 'high_20d_max', 'close_to_high_20d', 'is_new_high_20d', 'his_high_all', 'close_to_his_high', 'his_low_all', 'close_to_his_low', 'is_new_his_high']
            
            remain_list = list(dict.fromkeys(chain(basic_features, locals()[self.feature_type.value], advanced_features))) if self.feature_type != FeatureType.ALL else self.trade_df.columns.to_list()
            self.col_low, self.col_high, self.col_close = remain_list.index('low')-2, remain_list.index('high')-2, remain_list.index('close')-2 #在raw_data_np中的列索引位置,需要-2(减去ts_code和trade_date两列)

        #elif stock_type == StockType.RELATED:  # DELETE @ 20260102
        #    pass                               # DELETE @ 20260102
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
        # 由于要计算 t2，至少需要 3 天数据
        if self.trade_count < 3:
            logging.error("Not enough data to align trade dates.")
            return [],[],[],[]

        # 升序(old->new)对齐：以“能算出T2的T0”为基准
        self.__trade_datas     = self.__raw_data_pure_np[:-2, :]
        self.__trade_date_list = self.__trade_date_list[:-2]

        # t1 长度 N-1 -> 裁到 N-2
        self.__t1l_change_rate = self.__t1l_change_rate[:-1]
        self.__t1h_change_rate = self.__t1h_change_rate[:-1]

        # t2 长度 N-2 -> 刚好
        # self.__t2l_change_rate = self.__t2l_change_rate
        # self.__t2h_change_rate = self.__t2h_change_rate

        self.combine_data_np = np.column_stack((
            self.__trade_date_list,
            self.__trade_datas,
            self.__t1l_change_rate,
            self.__t1h_change_rate,
            self.__t2l_change_rate,
            self.__t2h_change_rate,
        ))
        self.y_cnt = 4

        # 3) 裁剪冷启动期：升序下应该裁掉头部 max_cut_days
        #    combine_data_np：训练用，裁头部即可（尾部2天已在上面[:-2]去掉）
        combine_out = self.combine_data_np[max_cut_days:]

        # 4) raw_data_np：预测用，保留最新 -> 只裁头部，不裁尾部
        raw_out = self.raw_data_np[max_cut_days:]

        return combine_out, raw_out


    #计算t1,t2变化率
    def update_t1_change_rate(self):
        """
        升序 (old->new) 下：
        T0 = i
        T1 = i+1

        t1l_change_rate[i] = (T1_low[i+1]  - T0_close[i]) / T0_close[i]
        t1h_change_rate[i] = (T1_high[i+1] - T0_close[i]) / T0_close[i]
        """
        if self.trade_count < 2:
            logging.error("Not enough data to calculate T1 change rate.")
            return
        
        if self.stock_type == StockType.INDEX:
            self.__t1l_change_rate = self.__raw_data_pure_np[:-1, 0] - self.__raw_data_pure_np[:-1, 0]  # INDEX没有涨跌幅，变化率为0
            self.__t1h_change_rate = self.__raw_data_pure_np[:-1, 0] - self.__raw_data_pure_np[:-1, 0]  # INDEX没有涨跌幅，变化率为0
            return
        # 计算T1变化率
        close_t0 = self.__raw_data_pure_np[:-1, self.col_close]
        low_t1   = self.__raw_data_pure_np[1:,  self.col_low]
        high_t1  = self.__raw_data_pure_np[1:,  self.col_high]

        self.__t1l_change_rate = (low_t1  - close_t0) / close_t0
        self.__t1h_change_rate = (high_t1 - close_t0) / close_t0

    
    def update_t2_change_rate(self):
        """
        升序 (old->new) 下：
        T0 = i
        T2 = i+2

        t2l_change_rate[i] = (T2_low[i+2]  - T0_close[i]) / T0_close[i]
        t2h_change_rate[i] = (T2_high[i+2] - T0_close[i]) / T0_close[i]
        """
        if self.trade_count < 3:
            logging.error("Not enough data to calculate T2 change rate.")
            return
        if self.stock_type == StockType.INDEX:
            self.__t2l_change_rate = self.__raw_data_pure_np[:-2, 0] - self.__raw_data_pure_np[:-2, 0]  # INDEX没有涨跌幅，变化率为0
            self.__t2h_change_rate = self.__raw_data_pure_np[:-2, 0] - self.__raw_data_pure_np[:-2, 0]  # INDEX没有涨跌幅，变化率为0
            return
        # 计算T2变化率
        close_t0 = self.__raw_data_pure_np[:-2, self.col_close]
        low_t2   = self.__raw_data_pure_np[2:,  self.col_low]
        high_t2  = self.__raw_data_pure_np[2:,  self.col_high]

        self.__t2l_change_rate = (low_t2  - close_t0) / close_t0
        self.__t2h_change_rate = (high_t2 - close_t0) / close_t0

if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    #ts_code = '000001.SH'
    ts_code = '600036.SH'
    t = Trade(ts_code, si, stock_type=StockType.RELATED, start_date='20250104', end_date='20250830')
    #t = Trade(ts_code, si, stock_type=StockType.INDEX, start_date='20250101', end_date='20250829')
    print(f"featurn names: {t.remain_list}")
    print(f"trade shape: {t.combine_data_np.shape}, raw shape: {t.raw_data_np.shape}")
    print(f"raw_data_np head:\n{pd.DataFrame(t.raw_data_np).head(5)}\nraw_data_np tail:\n{pd.DataFrame(t.raw_data_np).tail(5)}")
    print(f"combine_data_np head:\n{pd.DataFrame(t.combine_data_np).head(5)}\ncombine_data_np tail:\n{pd.DataFrame(t.combine_data_np).tail(5)}")
    