#-*- coding:UTF-8 -*-
import sys,os,time
import pandas as pd
from datetime import datetime
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stock import Stock
from utils.const_def import BASE
pd.set_option('future.no_silent_downcasting', True)

class Stocks():
    def __init__(self, ts_codes, stock_info, if_force_download=False, start_date='19910101', end_date=datetime.today().strftime('%Y%m%d')):
        self.stock_list, self.ts_code_list = [], []
        self.si = stock_info
        self.next_empty_trade_date = int(datetime.today().strftime('%Y%m%d'))
        self.start_date, self.end_date = start_date, end_date
        self.update_stock_data(ts_codes, if_force_download, start_date, end_date)
        self.update_next_empty_trade_date()
        print("INFO: total stock list count is < %d >."%len(self.stock_list))
        
        if int(self.next_empty_trade_date) <= int(self.end_date):   #文件数据不匹配要求的最新日期，需要更新数据
            print("INFO: stock data not fresh, need to dowload from tushare.")
            self.update_to_someday(self.end_date)
        else:
            self.save_all()
    
    #根据ts_code列表，逐个更新对应的股票数据，来源为csv文件或tushare（优先文件，无文件则从tushare读取）
    #遇到更新异常的情况，会自动重新处理
    def update_stock_data(self, ts_codes, if_force_download, start_date, end_date):
        remain_ts_code_list = ts_codes
        total_len = len(remain_ts_code_list)
        finished_cnt = 0
        while(len(remain_ts_code_list)>0):
            ts_code = remain_ts_code_list.pop(0)
            start_time = time.perf_counter()
            print("INFO: update stock - <%s>"%(ts_code))
            self.stock_list.append(Stock(ts_code, self.si, if_force_download=if_force_download, start_date=start_date, end_date=end_date))
            #    except Exception as e:
            #        print("ERROR: error msg <%s>"%e)
            #        remain_ts_code_list.append(ts_code)
            #        print("ERROR: ####################################################### update stock - <%s> failed."%(ts_code))
            finished_cnt += 1
            end_time = time.perf_counter()
            print("INFO: [%d/%d(%.1f%%)] finished in[%.1f]seconds. total line is <%d>\n"\
                  %(finished_cnt,total_len,finished_cnt/total_len*100,(end_time-start_time),self.stock_list[-1].df.shape[0]))
        self.ts_code_list = self.get_stock_list()
        print("INFO: update stock data done!\n")
    
    def update_next_empty_trade_date(self):
        for stock in self.stock_list:   #找到所有记录中，最后数据日期最早的那一天
            if self.next_empty_trade_date > int(stock.latest_date):
                self.next_empty_trade_date = int(stock.latest_date)
        self.next_empty_trade_date = self.si.get_next_trade_date(str(self.next_empty_trade_date))

    #返回股票代码列表清单
    def get_stock_list(self):
        ts_code_list = []
        if self.stock_list is None or len(self.stock_list) == 0:
            print("ERROR: get_stock_list() : <stock_list> is empty.")
            exit()
        for stock in self.stock_list:
            ts_code_list.append(stock.ts_code)
        return ts_code_list

    #存储所有的股票数据
    def save_all(self, base_dir=BASE):
        for stock in self.stock_list:
            stock.save(base_dir=base_dir)

    #获取特定日期的数据
    def get_spec_date_detail(self, date):
        df = self.si.get_stock_detail(spec_date=date)
        if len(self.stock_list)>0 :
            df = df[df['ts_code'].isin(self.ts_code_list)]
        else:
            print("ERROR: <stock_list> is empty.")
            exit()
        return df
    
    #更新特定日期的数据
    def update_daily(self, date, if_load=True, if_save=True):
        daily_df = self.get_spec_date_detail(date)
        for stock in self.stock_list:
            print("INFO: start update <%s(%s)>"%(stock.name, stock.ts_code))
            stock.update_daily_file(daily_df, date, if_load=if_load, if_save=if_save)

    #更新数据至特定日期
    def update_to_someday(self, to_date):
        need_update_dates = self.si.get_trade_open_dates(str(self.next_empty_trade_date), to_date).values
        print("INFO: days need to update - <%s>"%list(need_update_dates))
        if len(need_update_dates) == 0:
            print("INFO: date is fresh, no need to update")
        elif len(need_update_dates) > 1000:
            #未更新天数太多，直接按股票全量下载
            print("INFO: too many need update dates, will download all data.")
            #self.build_from_tushare(start_date=str(LATEST_DATE), end_date=to_date)
            self.__init__(self.ts_code_list, self.si, asset='E', if_force_download=True, start_date=self.start_date, end_date=to_date)
        else:
            #按天逐日更新
            need_update_days_cnt = len(need_update_dates)
            for date in reversed(need_update_dates):  #反向循环 即可实现从最早日期开始更新
                cur_update_date = date[0]
                print("\nINFO: update daily data for date - <%s>"%cur_update_date)
                if need_update_days_cnt > 1:    #更新多天
                    if cur_update_date == need_update_dates[0]:#最后一天，只存不读
                        self.update_daily(cur_update_date, if_load=False, if_save=True)
                    else:#其他的天数，不读不存
                        self.update_daily(cur_update_date, if_load=False, if_save=False)
                else:#只更新一天
                    self.update_daily(cur_update_date, if_load=False, if_save=True)

    #从给出的股票代码列表中，按市值排序并返回
    def get_sort_by_mv(self, stock_list):
        ret = self.si.filtered_list_df[self.si.filtered_list_df['ts_code'].isin(stock_list)]
        return ret.sort_values(by=['total_mv'], ascending=False)['ts_code'].values.tolist()