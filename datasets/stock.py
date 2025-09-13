#-*- coding:UTF-8 -*-
import sys,os,logging
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stockinfo import StockInfo
from utils.const_def import BASE_DIR, STOCK_DIR, INDEX_DIR
from utils.const_def import STOCK_DATE_DELTA
from utils.tk import TOKEN
from utils.utils import setup_logging
pd.set_option('future.no_silent_downcasting', True)

#用于从tushare读取/存储一只股票/指数的所有所需数据
class Stock():
    #si即stockinfo对象,必须传入
    def __init__(self, ts_code, si=None, start_date=None, end_date=None):
        self.ts_code = ts_code
        self.si = si
        self.asset = self.si.get_asset(self.ts_code)
        self.start_date = si.get_recent_trade_date(start_date) if start_date is not None else (self.si.get_start_date(self.ts_code, self.asset))
        self.start_date = self.start_date if (self.si.get_start_date(self.ts_code, self.asset)) < int(self.start_date) else (self.si.get_start_date(self.ts_code, self.asset)) 
        self.end_date = si.get_recent_trade_date(end_date) if end_date is not None else datetime.today().strftime('%Y%m%d')
        self.name = self.si.get_name(self.ts_code, self.asset)
        self.ffn = os.path.join(BASE_DIR, STOCK_DIR, self.ts_code + ".csv") if self.asset=='E' else os.path.join(BASE_DIR, INDEX_DIR, self.ts_code + ".csv")
        self.first_date, self.latest_date = 0, 0
        self.df_raw, self.df_filtered = None, None

        self.load()
        self.load(if_force_load_from_tushare=True) if self.if_need_re_update() else None
        self.df_filtered = self.get_df_between(self.start_date, self.end_date)
        #logging.info(f"[{self.name}({self.ts_code})]要求获取[{start_date}]-[{end_date}], 实际获取[{self.df_filtered['trade_date'].min()}]-[{self.df_filtered['trade_date'].max()}], 共<{self.df_filtered.shape[0]}>行")

    #根据csv文件是否存在,选择是从文件加载还是从tushare获取
    def load(self, if_force_load_from_tushare=False):
        if os.path.exists(self.ffn) and if_force_load_from_tushare==False:
            self.df_raw = pd.read_csv(self.ffn)
            self.df_raw['trade_date'] = self.df_raw['trade_date'].astype(np.int64)
            self.update_first_last_date()
            logging.info(f"[{self.name}({self.ts_code})]从<{self.ffn}>读取成功, <{self.df_raw.shape[0]}>行,[{self.first_date}]-[{self.latest_date}]")
        else:
            self.update_from_tushare()
            self.update_first_last_date()
            logging.info(f"[{self.name}({self.ts_code})]从tushare读取成功, 共<{self.df_raw.shape[0]}>行,[{self.first_date}]-[{self.latest_date}]")
            self.save()
        
    # 从tushare更新数据
    def update_from_tushare(self, start_date=None, end_date=None):
        s_date = start_date if start_date is not None else self.start_date
        logging.debug(f"[{self.name}({self.ts_code})]从tushare获取数据, start_date:{self.start_date}({type(self.start_date)}), end_date:{end_date}")
        if end_date is None:
            e_date = datetime.today().strftime('%Y%m%d')
        else:
            e_date = str(end_date)
        logging.debug(f"[{self.name}({self.ts_code})]从tushare获取数据[{s_date}]-[{e_date}]")
        delta = datetime.strptime(e_date,'%Y%m%d') - datetime.strptime(str(s_date),'%Y%m%d')
        if delta.days > STOCK_DATE_DELTA:
            logging.debug(f"一次需要读取的数据太多, 改为两次读取")
            m_date = (datetime.strptime(e_date,'%Y%m%d') - timedelta(days=STOCK_DATE_DELTA)).strftime("%Y%m%d")
            self.df_raw = self.si.get_stock_detail(asset=self.asset, ts_code=self.ts_code, start_date=m_date, end_date=e_date)
            m_date = (datetime.strptime(e_date,'%Y%m%d') - timedelta(days=STOCK_DATE_DELTA + 1)).strftime("%Y%m%d")
            new_df = self.si.get_stock_detail(asset=self.asset, ts_code=self.ts_code, start_date=s_date, end_date=m_date)
            self.df_raw = pd.concat([self.df_raw.astype(new_df.dtypes), new_df.astype(self.df_raw.dtypes)])   #due to padas upgrade
            self.df_raw = self.df_raw.reset_index(drop=True)
        else:   
            logging.debug("可一次获取完全部数据.")
            self.df_raw = self.si.get_stock_detail(asset=self.asset, ts_code=self.ts_code, start_date=s_date, end_date=e_date)
        self.update_first_last_date()
        logging.debug(f"[{self.name}({self.ts_code})]加载数据[{s_date}]-[{e_date}]成功, 共<{self.df_raw.shape[0]}>行, 最新日期<{self.latest_date}>")

    #判断是否需要重新下载数据
    #如果最新日期小于要求的结束日期,或者最早日期大于要求的开始日期,则需要重新下载
    def if_need_re_update(self):
        logging.debug(f"self.latest_date:{self.latest_date}({type(self.latest_date)}), self.first_date:{self.first_date}({type(self.first_date)}), self.start_date:{self.start_date}({type(self.start_date)}), self.end_date:{self.end_date}({type(self.end_date)})")
        ret = int(self.latest_date) < int(self.end_date) or int(self.first_date) > int(self.start_date)
        if ret:
            logging.debug(f"[{self.name}({self.ts_code})]需要重新更新数据, 文件最后日期<{self.latest_date}>, 要求结束日期<{self.end_date}>, 文件最早日期<{self.first_date}>, 要求开始日期<{self.start_date}>")
        return ret
    
    #获取给定日期的数据
    def get_df_between(self, start_date, end_date):
        if self.df_raw is not None:
            mask = (self.df_raw['trade_date'].astype(int) >= int(start_date)) & (self.df_raw['trade_date'].astype(int) <= int(end_date))
            #logging.info(f"[{self.name}({self.ts_code})]要求获取[{start_date}]-[{end_date}], 实际获取[{self.df_raw[mask]['trade_date'].min()}]-[{self.df_raw[mask]['trade_date'].max()}], 共<{self.df_raw[mask].shape[0]}>行")
            return self.df_raw[mask]
        else:
            logging.error(f"[{self.name}({self.ts_code})] - 数据为空，无法获取指定日期范围的数据.")
            return pd.DataFrame()

    def update_first_last_date(self):
        if self.df_raw is not None:
            self.first_date, self.latest_date = int(self.df_raw['trade_date'].min()), int(self.df_raw['trade_date'].max())
            logging.debug(f"[{self.name}({self.ts_code})] - first_date:{self.first_date}({type(self.first_date)}), latest_date:{self.latest_date}({type(self.latest_date)})")
        else:
            logging.error(f"[{self.name}({self.ts_code})] - 数据为空，无法更新日期.")
            exit()

    def save(self):
        logging.debug(f"[{self.name}({self.ts_code})]写入文件 - [{self.ffn}]")
        if self.df_raw is not None:
            self.df_raw.to_csv(self.ffn, index=False)
            logging.info(f"[{self.name}({self.ts_code})]写入文件 - [{self.ffn}]成功, <{self.df_raw.shape[0]}>行")
        else:
            logging.error(f"[{self.name}({self.ts_code})]写入文件 - <{self.ffn}>失败, df为空")
            exit()
    
    #在股票的df中,写入指定日期的日线数据
    def update_daily_file(self, new_df, date, if_load=True, if_save=True):
        if if_load:
            self.load()
        if new_df is None:
            logging.error(f"[{self.name}({self.ts_code})] - 函数输入错误 - <new_df> is None.")
            exit()
        if new_df.shape[0] == 0:
            logging.info(f"[{self.name}({self.ts_code})]在[{date}]的数据为空, 将跳过此天不更新.")
        else:
            if self.df_raw[self.df_raw['trade_date'].isin([int(date)])].shape[0] == 0:  #还没有此日期的数据，可以添加
                self.df_raw = pd.concat([new_df[new_df['ts_code']==self.ts_code], self.df_raw], ignore_index=True)
            else:   
                logging.info(f"[{self.name}({self.ts_code})]在[{date}]的数据已经存在, 将跳过此天不更新.")
        if if_save:
            self.save()


    
if __name__ == "__main__":
    setup_logging()
    si = StockInfo(TOKEN)
    ts_code = '399001.SZ'

    s = Stock(ts_code, si, start_date='20250101', end_date='20250826')
    #s = Stock(ts_code, si)

    #s.update_daliy()
    #ss = Stocks([], ti)
    #new_df = ss.get_spec_date_detail('20250822')
    #df.to_csv('data\\temp\\20250814.csv')

    #s.update_daily_file(new_df)
    #print(s.df)
