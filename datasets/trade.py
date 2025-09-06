#-*- coding:UTF-8 -*-
import sys,os,logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
from pathlib import Path

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stock import Stock
from stockinfo import StockInfo
from utils.utils import StockType, setup_logging
from utils.tk import TOKEN
from utils.const_def import BASE_DIR, TMP_DIR

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
    def __init__(self, ts_code, si, stock_type=StockType.PRIMARY, start_date=None, end_date=None):
        self.ts_code, self.si = ts_code, si
        self.start_date = start_date if start_date is not None else str(self.si.get_start_date(self.ts_code))
        self.end_date = end_date if end_date is not None else datetime.today().strftime('%Y%m%d')
        self.asset = self.si.get_asset(self.ts_code)
        self.stock_type = stock_type if self.asset == 'E' else StockType.INDEX
        logging.debug(f"Trade::__init__() - ts_code:{ts_code}, start_date:{self.start_date}, end_date:{self.end_date}")
        self.stock = Stock(ts_code, si, self.start_date, self.end_date)
        self.trade_count = len(self.stock.df_filtered['trade_date'].values)
        logging.debug(f"[{self.stock.name}({self.ts_code})]交易数据行数:<{self.trade_count}>")

        #0. 获取原始数据,所有的特征都需要通过此原始数据计算得到
        self.raw_data_df = self.stock.df_filtered.copy().reset_index(drop=True)  #原始数据的DataFrame格式
        
        #1. 增删特征数据(增删对象为self.trade_df), 并返回新增特征需要丢弃的天数
        max_cut_days = self.update_new_feature()  #新增特征数据,并返回新增特征需要丢弃的天数

        #1.5 根据数据类型,删除不需要的特征
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

        #logging.debug(f"combine_data_np head -\n{pd.DataFrame(self.combine_data_np).head(5)}\ncombine_data_np tail -\n{pd.DataFrame(self.combine_data_np).tail(5)}")
        #logging.debug(f"raw_data_np head -\n{pd.DataFrame(self.raw_data_np).head(5)}\nraw_data_np tail -\n{pd.DataFrame(self.raw_data_np).tail(5)}")
        logging.info(f"[{self.stock.name}({self.ts_code})]最终可用数据行数:<{self.__trade_datas.shape[0]}>，特征列数:<{self.raw_data_np.shape[1]-1}>")

    #新增特征列
    def update_new_feature(self):
        max_cut_days = 0
        #1. 创建一个交易数据的DataFrame格式,并按日期升序排列(最早日期在前),TA-lib需要按升序排列
        self.trade_df = self.raw_data_df.copy().sort_index(ascending=False).reset_index(drop=True) #按日期升序排列,方便计算

        #2. 通过计算,新增特征数据(新增列)
        #5个基本的技术指标
        self.trade_df['rsi_14'], max_cut_days = ta.rsi(self.trade_df['close'], length=14), max(max_cut_days, 13)
        self.trade_df['macd'], self.trade_df['macd_signal'], self.trade_df['macd_hist'] = ta.macd(self.trade_df['close'])['MACD_12_26_9'], ta.macd(self.trade_df['close'])['MACDs_12_26_9'], ta.macd(self.trade_df['close'])['MACDh_12_26_9']
        max_cut_days = max(max_cut_days, 25+8)  #macd需要25天数据才能计算出来
        self.trade_df['atr_14'], max_cut_days = ta.atr(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=14), max(max_cut_days, 14)
        self.trade_df['cci_20'], max_cut_days = ta.cci(self.trade_df['high'], self.trade_df['low'], self.trade_df['close'], length=20), max(max_cut_days, 19)
        self.trade_df, max_cut_days = self.trade_df.join(ta.bbands(self.trade_df['close'], length=20, std=2)), max(max_cut_days, 20)
        #补充日期\星期特征
        self.trade_df['date_mmdd'] = self.trade_df['trade_date'].astype(str).str[4:8].astype(int)
        self.trade_df['weekday'] = pd.to_datetime(self.trade_df['trade_date'], format='%Y%m%d').dt.weekday+1

        if True:   #以下特征效果不好,暂时不启用
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
        
        #3. 将trade_df按日期降序排列(最新日期在前),方便后续使用
        self.trade_df = self.trade_df.copy().sort_index(ascending=False).reset_index(drop=True) 
        return max_cut_days
    
    #根据数据类型,删除不需要的特征
    def drop_features_by_type(self, stock_type):
        #drop_list = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv', 'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',  'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',  'net_mf_vol', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr_14',  'cci_20', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0',  'BBP_20_2.0', 'date_mmdd', 'weekday']
        if stock_type == StockType.PRIMARY:
            #remain_list = self.trade_df.columns.to_list()
            #下面是皮尔逊筛选结果(上0.15多分类,下0.15回归)
            #remain_list = ['ts_code', 'trade_date', 'high', 'low', 'close', 'pb', 'dv_ratio', 'atr_14', 'BBB_20_2.0', 'natr_14']
            #remain_list = ['ts_code', 'trade_date', 'high', 'low', 'close', 'pe', 'pb', 'ps', 'dv_ratio', 'atr_14', 'BBB_20_2.0', 'obv', 'natr_14']
            #下面是互信息筛选结果(上0.03多分类,下0.03回归)
            #remain_list = ['ts_code', 'trade_date', 'high', 'low', 'close', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv', 'macd_signal', 'atr_14', 'BBL_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'obv', 'natr_14']
            #remain_list = ['ts_code', 'trade_date', 'high', 'low', 'close', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv', 'macd_signal', 'atr_14', 'BBL_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'obv', 'natr_14']
            #皮尔逊+互信息+树模型交集特征
            remain_list = ['ts_code', 'trade_date', 'high', 'low', 'close', 'sell_elg_vol', 'pb', 'obv', 'turnover_rate_f', 'dv_ratio', 'buy_sm_vol', 'close', 'stddev_10', 'natr_14', 'buy_md_vol', 'BBB_20_2.0', 'amount', 'atr_14']
            self.col_low, self.col_high, self.col_close = remain_list.index('low')-2, remain_list.index('high')-2, remain_list.index('close')-2
        elif stock_type == StockType.RELATED:
            remain_list = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'turnover_rate_f', 'volume_ratio', 'pe', 'pb', 'ps', 'dv_ratio', 'total_mv', 'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',  'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',  'net_mf_vol', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr_14',  'cci_20', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0',  'BBP_20_2.0']
        elif stock_type == StockType.INDEX:
            #remain_list = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol']
            remain_list = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pre_close']
        else:
            logging.error(f"Unknown stock type:{stock_type}, no features dropped!")
            return
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
        self.__t1_change_rate = self.__t1_change_rate[1:]    #从第二天开始对齐
        #self.t2_change_rate = self.t2_change_rate       #从第三天开始对齐,不需要变动
        
        #2. 组合生成头部对齐的数据
        self.combine_data_np = np.column_stack((self.__trade_date_list, self.__trade_datas, self.__t1_change_rate, self.__t2_change_rate))
        
        #3. 统一剪掉尾部并返回
        return self.combine_data_np[:-max_cut_days], self.raw_data_np[:-max_cut_days]

    #计算t1,t2变化率
    #t1_change_rate表示T1低值的变化率
    def update_t1_change_rate(self):#表示T1低值的变化率
        ##计算说明:t1_change_rate = t1_low - t0_close / t0_close
        if self.trade_count < 2:
            logging.error("Not enough data to calculate T1 change rate.")
            return
        self.__t1_change_rate = (self.__raw_data_pure_np[:-1, self.col_low] - self.__raw_data_pure_np[1:, self.col_close]) / self.__raw_data_pure_np[1:, self.col_close] if self.stock_type == StockType.PRIMARY else self.__raw_data_pure_np[:-1, 0]-self.__raw_data_pure_np[:-1, 0]
        #self.t1_change_rate = np.array([RateCat(rate=x,scale=T1L_SCALE).get_label() for x in self.t1_change_rate])
    
    #t2_change_rate表示T2高值的变化率
    def update_t2_change_rate(self):#表示T2高值的变化率
        ##计算说明:t2_change_rate = t2_high - t0_close / t0_close
        if self.trade_count < 3:
            logging.error("Not enough data to calculate T2 change rate.")
            return
        self.__t2_change_rate = (self.__raw_data_pure_np[:-2, self.col_high] - self.__raw_data_pure_np[2:, self.col_close]) / self.__raw_data_pure_np[2:, self.col_close] if self.stock_type == StockType.PRIMARY else self.__raw_data_pure_np[:-2, 0]-self.__raw_data_pure_np[:-2, 0]
        #self.t2_change_rate = np.array([RateCat(rate=x,scale=T2H_SCALE).get_label() for x in self.t2_change_rate])


if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    #ts_code = '000001.SH'
    ts_code = '600036.SH'
    t = Trade(ts_code, si, stock_type=StockType.RELATED, start_date='20250701', end_date='20250829')
    #t.print_combine_data(t.trade_datas_combine, "ALL")
    #t.save_combine_data()
