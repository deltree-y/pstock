#-*- coding:UTF-8 -*-
import sys,os,time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stock import Stocks
from stockinfo import StockInfo
from utils.const_def import BASE, TOKEN, LATEST_DATE
from utils.const_def import T1_LOWER, T1_UPPER, T2_LOWER, T2_UPPER

#数据说明：
#下面数据均行数均已对齐，每个有t1t2涨跌幅的交易日为一行 - 
#       self.trade_datas           不含日期，可直接使用的所有交易数据
#       self.trade_datas_with_mark 不含日期，与self.trade_datas的唯一区别是最后一列为是否符合要求的bool值
#       self.trade_datas_combine   包含日期(首列)，t1t2变化率(最后两列), 不可直接使用的所有交易数据
#       self.trade_date_list       所有交易日期列表
#       self.fit_mark              所有交易数据的标记，符合要求的为True, 不符合要求的为False
#下面数据未与可训练数据对齐，使用时应注意 
#       self.fit_datas             仅包含达到涨跌幅的交易数据，格式同self.trade_datas_combine
class Trade():
    def __init__(self, stock, t1_rate_lower=T1_LOWER, t1_rate_upper=T1_UPPER, t2_rate_lower=T2_LOWER, t2_rate_upper=T2_UPPER):
        self.stock = stock
        self.t1_rate_lower, self.t1_rate_upper = t1_rate_lower, t1_rate_upper
        self.t2_rate_lower, self.t2_rate_upper = t2_rate_lower, t2_rate_upper
        self.raw_data_df = stock.df
        self.raw_data_np = stock.df.to_numpy()
        self.raw_data_pure = np.delete(self.raw_data_np, [0,1], axis=1)  #删除前两列ts_code,name
        self.trade_date_list = stock.df['trade_date'].values    #只保存有t1，t2数据的交易日
        self.trade_count = len(self.trade_date_list)
        self.t1_change_rate = self.get_t1_change_rate()
        self.t2_change_rate = self.get_t2_change_rate()
        self.trade_datas, self.trade_date_list, self.t1_change_rate, self.trade_datas_combine = self.get_aligned_trade_dates()  
        self.fit_datas, self.fit_mark = self.get_fit_datas()

        #self.print_combine_data(self.trade_datas_combine, "ALL")
        #self.print_combine_data(self.fit_datas, "FIT")

    def get_t1_change_rate(self):#表示T1低值的变化率
        #t1_change_rate = t1_low - t0_close / t0_close
        if self.trade_count < 2:
            print("ERROR: Trade() - Not enough data to calculate T1 change rate.")
            return
        self.t1_change_rate = (self.raw_data_pure[:-1, 2] - self.raw_data_pure[1:, 3]) / self.raw_data_pure[1:, 3]
        #for date,t1_change_rate in zip(self.trade_date_list[1:], self.t1_change_rate[:]):
        #    print("DEBUG: trade date:<%s>, t1 change rate:<%+.2f%%>"%(date,t1_change_rate*100))
        return self.t1_change_rate

    def get_t2_change_rate(self):#表示T2高值的变化率
        #t2_change_rate = t2_high - t0_close / t0_close
        if self.trade_count < 3:
            print("ERROR: Trade() - Not enough data to calculate T2 change rate.")
            return
        self.t2_change_rate = (self.raw_data_pure[:-2, 1] - self.raw_data_pure[2:, 3]) / self.raw_data_pure[2:, 3]
        #for date,t2_change_rate in zip(self.trade_date_list[2:], self.t2_change_rate[:]):
        #    print("DEBUG: trade date:<%s>, t2 change rate:<%+.2f%%>"%(date,t2_change_rate*100))
        return self.t2_change_rate

    def get_aligned_trade_dates(self):
        if self.trade_count < 2:
            print("ERROR: Trade() - Not enough data to align trade dates.")
            return [],[],[],[]
        # Align trade dates with T1 and T2 change rates
        self.trade_datas = self.raw_data_pure[2:, :]  #从第三天开始对齐
        self.trade_date_list = self.trade_date_list[2:]
        self.t1_change_rate = self.t1_change_rate[1:]
        #self.t2_change_rate = self.t2_change_rate
        self.trade_datas_combine = np.column_stack((self.trade_date_list, self.trade_datas, self.t1_change_rate, self.t2_change_rate))
        return self.trade_datas, self.trade_date_list, self.t1_change_rate, self.trade_datas_combine

    def get_fit_datas(self):
        if len(self.trade_datas_combine) == 0:
            print("ERROR: Trade() - No trade data available for fitting.")
            return [],[]
        # Filter trade data based on T1 and T2 change rates
        fit_datas = self.trade_datas_combine[
            (self.trade_datas_combine[:, -2] >= self.t1_rate_lower) & (self.trade_datas_combine[:, -2] <= self.t1_rate_upper) &
            (self.trade_datas_combine[:, -1] >= self.t2_rate_lower) & (self.trade_datas_combine[:, -1] <= self.t2_rate_upper)
            ]
        #print(fit_datas)
        self.fit_mark = (self.trade_datas_combine[:, -2] >= self.t1_rate_lower) & (self.trade_datas_combine[:, -2] <= self.t1_rate_upper) &\
                        (self.trade_datas_combine[:, -1] >= self.t2_rate_lower) & (self.trade_datas_combine[:, -1] <= self.t2_rate_upper) 

        if len(fit_datas) == 0:
            #print("ERROR: Trade() - No fitting data found.")
            fit_datas = []
        return fit_datas, self.fit_mark

    def print_combine_data(self, data, data_name):
        total_profit = 0
        print()
        print("INFO: %s data for <%s(%s)> ---"%(data_name,self.stock.name,self.stock.ts_code))

        if data is None or len(data) == 0:
            print("INFO: No %s data available to display."%data_name)
        else:
            for d in data:
                profit = (d[-1]-d[-2])*100
                total_profit += profit
                print("INFO: <%s>, h/l:<%.2f>/<%.2f>, close:<%.2f>, t1/t2:<%+.3f%%/%+.3f%%>, profit:[%.2f%%]" %
                        (d[0], d[2], d[3], d[4], d[-2]*100, d[-1]*100, (d[-1]-d[-2])*100))
        print("\n************************************************************************************")
        print("INFO: Total profit for <%s(%s)>: [%.2f%%]" % (self.stock.name, self.stock.ts_code, total_profit))

    def save_combine_data(self):
        df = pd.DataFrame(self.trade_datas_combine)
        df.columns = ['trade_date'] + list(self.raw_data_df.columns[2:]) + ['t1_change_rate', 't2_change_rate']
        df[['trade_date','open','high','low','close','pre_close','t1_change_rate', 't2_change_rate']]\
            .to_csv(BASE + "\\combine_data\\" + self.stock.ts_code + "_combine_data.csv", index=False)
    
if __name__ == "__main__":
    si = StockInfo(TOKEN)
    #download_list = si.get_filtered_stock_list(mmv=250000000)
    download_list = ['600036.SH']
    sse = Stocks(download_list, si, start_date='20020409', end_date='20250828', if_force_download=False)
    t = Trade(sse.stock_list[0])
    t.print_combine_data(t.trade_datas_combine, "ALL")
    #t.save_combine_data()
