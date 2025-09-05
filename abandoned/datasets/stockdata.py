# coding=utf-8
import csv
import pandas as pd
from pathlib import Path

import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.const_def import T1L_TV,T1H_TV,T2L_TV,T2H_TV,T3L_TV,T3H_TV
from utils.const_def import BASE
from utils.const_def import NUM_CLASSES
from utils.utils import get_datetime_date, day_plus_minor
from predict.cat import Cat
from abandoned.datasets.price import Price


class Data():
    def __init__(self):
        pass

    def get_unit(self):
        pass

#TradeData()为某个股票或指数的单个交易日数据集合
#           每个数据指标均使用变量进行命名
class TradeData():
    def __init__(self, asset='E', date=None, open=0, high=0, low=0, close=0, p_close=0, pct_chg=0, vol=0, turnover_rate=0, pe=0, pb=0, ps=0, total_share=0, float_share=0, 
            buy_sm_vol=0, sell_sm_vol=0, buy_md_vol=0, sell_md_vol=0, buy_lg_vol=0, sell_lg_vol=0, buy_elg_vol=0, sell_elg_vol=0,net_mf_vol=0):
        self.num_classes = NUM_CLASSES
        self.asset = asset
        self.date = date
        self.open = float(open)
        self.high = float(high) 
        self.low = float(low)
        self.close = float(close)
        self.p_close = float(p_close)
        self.p = Price(self.open, self.close, self.low, self.high)
        self.pct_chg = 0 if pct_chg == '' else float(pct_chg)
        self.vol = float(vol)
        self.turnover_rate = 0 if turnover_rate == '' else float(turnover_rate)
        self.pe = 0 if pe == '' else float(pe)
        self.pb = 0 if pb == '' else float(pb)
        self.ps = 0 if ps == '' else float(ps)
        self.total_share = 0 if total_share == '' else float(total_share)
        self.float_share = 0 if float_share == '' else float(float_share)
        self.buy_sm_vol = 0 if buy_sm_vol == '' else float(buy_sm_vol)
        self.sell_sm_vol = 0 if sell_sm_vol == '' else float(sell_sm_vol)
        self.buy_md_vol = 0 if buy_md_vol == '' else float(buy_md_vol)
        self.sell_md_vol = 0 if sell_md_vol == '' else float(sell_md_vol)
        self.buy_lg_vol = 0 if buy_lg_vol == '' else float(buy_lg_vol)
        self.sell_lg_vol = 0 if sell_lg_vol == '' else float(sell_lg_vol)
        self.buy_elg_vol = 0 if buy_elg_vol == '' else float(buy_elg_vol)
        self.sell_elg_vol = 0 if sell_elg_vol == '' else float(sell_elg_vol)
        self.net_mf_vol = 0 if net_mf_vol == '' else float(net_mf_vol)
        #self.pct_chg_categorical = Cat(pct_chg=self.pct_chg, TV=TV).get_cat()   #将变化率转换为模型输入对应的0/1列表

        self.t1_low,self.t1_high, self.t2_low, self.t2_high, self.t3_low, self.t3_high= 0,0,0,0,0,0
        self.t1_low_chg_pct, self.t1_high_chg_pct, self.t2_low_chg_pct, self.t2_high_chg_pct, self.t3_low_chg_pct, self.t3_high_chg_pct = 0,0,0,0,0,0
        self.t1_low_categorical, self.t1_high_categorical, self.t2_low_categorical, self.t2_high_categorical, self.t3_low_categorical, self.t3_high_categorical = [],[],[],[],[],[]

    def set_empty_data(self):
        self.open, self.high, self.low, self.close, self.p_close, self.pct_chg = -1,-1,-1,-1,-1,-1
        self.vol, self.turnover_rate, self.pe, self.pb, self.ps, self.total_share, self.float_share = -1,-1,-1,-1,-1,-1,-1
        self.buy_sm_vol, self.buy_md_vol, self.buy_lg_vol, self.buy_elg_vol  = -1,-1,-1,-1
        self.sell_sm_vol,self.sell_md_vol,self.sell_lg_vol,self.sell_elg_vol = -1,-1,-1,-1
        self.t1_low, self.t1_high, self.t2_low, self.t2_high, self.t3_low, self.t3_high = -1,-1,-1,-1,-1,-1


#StockData用于存储某一只股票或指数的所有交易日数据
#入参ts_code，则自动读入data目录下以该code命名的csv文件中的数据
#首行数据为最近的日期对应数据
#其中：
#   trade_datas - (TradeData类型)列表，用于存储所有的交易日数据，每个交易日数据为一个元素
class StockData(Data):
    def __init__(self, ts_code, asset='E', adj='None'):
        super(StockData, self).__init__()
        self.ts_code = ts_code
        self.asset = asset
        self.adj = adj
        #self.stock = Stock(ts_code, asset=self.asset, adj=self.adj)
        self.trade_datas = []   #TradeData()类型
        self.trade_data_cnt = 0
        self.fn = BASE + self.ts_code + ".csv"

        self.read_stock_file()
        self.build_raise_down_data()
        self.end_date = self.trade_datas[0].date
        self.start_date = self.trade_datas[self.trade_data_cnt-1].date
        print("INFO: [%s] start/end date - <%s>/<%s>"%(self.ts_code, self.start_date, self.end_date))

    #从csv文件读入所有交易数据
    def read_stock_file(self):
        if self.fn is not None:
            print("INFO: open <%s>..."%self.fn, end="")
            reader = csv.reader(open(self.fn, "r"))
        else:
            print("ERROR: open stock file - <%s> failed."%self.fn)
            return -1
        
        for item in reader:
            if reader.line_num > 1:
                if self.asset == 'E':
                    self.trade_datas.append(TradeData('E',item[2],item[3],item[4],item[5],item[6],item[7],item[9],item[10],item[14],
                                                      item[17],item[19],item[20],item[24],item[25],item[30],item[32],item[34],
                                                      item[36],item[38],item[40],item[42],item[44],item[46]))
                elif self.asset == 'I':
                    self.trade_datas.append(TradeData('I',item[2],item[3],item[4],item[5],item[6],item[7],item[9],item[10]))
                else:
                    print("\nERROR: StockData()::read_stock_file() error.")
                    pass
        self.trade_data_cnt = reader.line_num - 1
        print("complete. item cnt-<%d>, "%self.trade_data_cnt, end="")

    #根据输入的日期，返回对应日期的TradeData数据
    #若没找到对应日期的数据的话，会返回上一个交易日的数据
    #若查找日期比首个交易日还早的话，则返回第一个交易日的数据
    def get_data_by_date(self, trade_date):
        empty_data = TradeData(asset='E', date=trade_date)
        empty_data.set_empty_data()
        if self.is_earlier_than_start_date(trade_date): #若查找日期比首个交易日还早的话，则返回第一个交易日的数据
            #print("WARNING: StockData.get_data_by_date() - earlier than start date. code:<%s>, date:<%s>, will get first date's data."%(self.ts_code, trade_date))
            #print("WARNING: StockData.get_data_by_date() - earlier than start date. code:<%s>, date:<%s>, will get empty data."%(self.ts_code, trade_date))
            #return TradeData(date=self.start_date)  
            return empty_data  #返回一个全-1的集合
        else:
            for trade_data in self.trade_datas:
                if trade_data.date == trade_date:   #如果找到对应日期的数据，则直接返回
                    return trade_data
            #print("INFO: StockData.get_data_by_date() - no data found code:<%s>, date:<%s>, will get previous date's data."%(self.ts_code, trade_date))
            return empty_data      #没找到对应日期的数据的话，会返回上一个全-1的集合
            #return self.get_data_by_date(get_minor_one_day(trade_date))   #没找到对应日期的数据的话，会返回上一个交易日的数据

    #根据输入的日期，返回对应日期的TradeData数据
    #若没找到对应日期的数据的话，返回None
    def get_exact_data_by_date(self, trade_date):
        for trade_data in self.trade_datas:
            if trade_data.date == trade_date:
                return trade_data
        #print("WARNING: StockData.get_exact_data_by_date() - no data found code:<%s>, date:<%s>, will return None."%(self.ts_code, trade_date))
        return None

    #根据输入的日期，返回对应日期的下一个交易日datetime
    #如果没找到，则退出
    def get_next_trade_date(self, trade_date):
        i = 1
        while i < 50:
            t1 = day_plus_minor(trade_date,i)
            for trade_data in self.trade_datas:
                if trade_data.date == t1:
                    return t1
            i += 1
        print("ERROR: StockData.get_next_trade_date() - no next date found. code:<%s>, date:<%s>, will exit."%(self.ts_code, trade_date))
        sys.exit()



    def is_earlier_than_start_date(self, string_date):
        return get_datetime_date(string_date)<get_datetime_date(self.start_date)
    
    #生成T1,T2,T3涨跌幅数据，用于预测
    def build_raise_down_data(self):
        print("")
        for index in range(3, self.trade_data_cnt):
            self.trade_datas[index].t1_low = self.trade_datas[index-1].low
            self.trade_datas[index].t1_high = self.trade_datas[index-1].high
            self.trade_datas[index].t1_low_chg_pct = 100*(self.trade_datas[index].t1_low-self.trade_datas[index].close)/self.trade_datas[index].close
            self.trade_datas[index].t1_high_chg_pct = 100*(self.trade_datas[index].t1_high-self.trade_datas[index].close)/self.trade_datas[index].close
            self.trade_datas[index].t1_low_categorical = Cat(pct_chg=self.trade_datas[index].t1_low_chg_pct, TV=T1L_TV).get_cat()
            self.trade_datas[index].t1_high_categorical = Cat(pct_chg=self.trade_datas[index].t1_high_chg_pct, TV=T1H_TV).get_cat()

            self.trade_datas[index].t2_low = self.trade_datas[index-2].low
            self.trade_datas[index].t2_high = self.trade_datas[index-2].high
            self.trade_datas[index].t2_low_chg_pct = 100*(self.trade_datas[index].t2_low-self.trade_datas[index].close)/self.trade_datas[index].close
            self.trade_datas[index].t2_high_chg_pct = 100*(self.trade_datas[index].t2_high-self.trade_datas[index].close)/self.trade_datas[index].close
            self.trade_datas[index].t2_low_categorical = Cat(pct_chg=self.trade_datas[index].t2_low_chg_pct, TV=T2L_TV).get_cat()
            self.trade_datas[index].t2_high_categorical = Cat(pct_chg=self.trade_datas[index].t2_high_chg_pct, TV=T2H_TV).get_cat()

            self.trade_datas[index].t3_low = self.trade_datas[index-3].low
            self.trade_datas[index].t3_high = self.trade_datas[index-3].high
            self.trade_datas[index].t3_low_chg_pct = 100*(self.trade_datas[index].t3_low-self.trade_datas[index].close)/self.trade_datas[index].close
            self.trade_datas[index].t3_high_chg_pct = 100*(self.trade_datas[index].t3_high-self.trade_datas[index].close)/self.trade_datas[index].close
            self.trade_datas[index].t3_low_categorical = Cat(pct_chg=self.trade_datas[index].t3_low_chg_pct, TV=T3L_TV).get_cat()
            self.trade_datas[index].t3_high_categorical = Cat(pct_chg=self.trade_datas[index].t3_high_chg_pct, TV=T3H_TV).get_cat()
            
            #print("DEBUG: <%s>. close<%.2f>. T1,2,3-<%.2f>(%.1f)/<%.2f>(%.1f),<%.2f>(%.1f)/<%.2f>(%.1f),<%.2f>(%.1f)/<%.2f>(%.1f)"%(self.trade_datas[index].date, self.trade_datas[index].close, \
            #                                                                                        self.trade_datas[index].t1_low, t1_low_chg_pct, \
            #                                                                                        self.trade_datas[index].t1_high, t1_high_chg_pct, \
            #                                                                                        self.trade_datas[index].t2_low, t2_low_chg_pct, \
            #                                                                                        self.trade_datas[index].t2_high, t2_high_chg_pct, \
            #                                                                                        self.trade_datas[index].t3_low, t3_low_chg_pct, \
            #                                                                                        self.trade_datas[index].t3_high, t3_high_chg_pct, ))

    
    #在指定日期的给定价格，是否可成功卖出。其中 -
    #   sp - sell price 卖出价格
    #   sd - sell date  卖出日期
    def is_good_sell_in(self, sp=None, sd=None):
        if any([sp is None, sd is None]):
            print("ERROR: StockData().is_good_sell_in() must input all parameters.")
            sys.exit()
        td = self.get_exact_data_by_date(sd)
        if td is None:  #未找到对应日期的数据
            print("ERROR: StockData().get_exact_data_by_date() data not found for date-<%s>."%str(sd))
            sys.exit()
        return td.p.is_good_sell(sp)

    #在指定日期的给定价格，是否可成功买入。其中 -
    #   bp - buy price 买入价格
    #   bd - buy date  买入日期
    def is_good_buy_in(self, bp=None, bd=None):
        if any([bp is None, bd is None]):
            print("ERROR: StockData().is_good_buy_in_days() must input all parameters.")
            sys.exit()
        td = self.get_exact_data_by_date(bd)
        if td is None:  #未找到对应日期的数据
            print("ERROR: StockData().get_exact_data_by_date() data not found for date-<%s>."%str(bd))
            sys.exit()
        return td.p.is_good_buy(bp)

    #在指定日期的给定价格，在后续cd天内是否可成功买入。其中 -
    #   bp  - buy price 买入价格
    #   sbd - start buy date  开始买入日期
    #   cd  - continued date  持续天数
    def is_good_buy_in_days(self, bp=None, sbd=None, cd=None):
        if any([bp is None, sbd is None, cd is None]):
            print("ERROR: StockData().is_good_buy_in_days() must input all parameters.")
            sys.exit()
        dd = 0  #date delta，用于计算日期差
        i = 0
        while(i<cd):
            date = day_plus_minor(sbd,dd)
            td = self.get_exact_data_by_date(date)
            dd += 1
            if get_datetime_date(date)>get_datetime_date(self.end_date):    #如果已找到最后一天，则退出循环，返回失败
                print("WARNING: StockData().is_good_buy_in_days cur date<%s> is later than end date<%s>."%((date),(self.end_date)))
                return False
            if td is None:  #未找到对应日期的数据, 继续找下一天，计数器不变
                continue
            else:   #对应日期有数据，则判断并确认是否返回
                #print("DEBUG: i-<%d>,  date-<%s> - %s"%(i,date, td.p.is_good_buy(bp)))
                if self.is_good_buy_in(bp, date):   #真值，直接返回
                    return True
                i += 1  #否则继续循环
        return False    #一直无真值，则直接返回失败
                
    #在指定日期的给定价格，在后续cd天内是否可成功卖出。其中 -
    #   sp  - sell price 卖出价格
    #   ssd - start sell date  开始卖出日期
    #   cd  - continued date  持续天数
    def is_good_sell_in_days(self, sp=None, ssd=None, cd=None):
        if any([sp is None, ssd is None, cd is None]):
            print("ERROR: StockData().is_good_sell_in_days() must input all parameters.")
            sys.exit()
        dd = 0  #date delta，用于计算日期差
        i = 0
        while(i<cd):
            date = day_plus_minor(ssd,dd)
            td = self.get_exact_data_by_date(date)
            dd += 1
            if get_datetime_date(date)>get_datetime_date(self.end_date):    #如果已找到最后一天，则退出循环，返回失败
                print("WARNING: StockData().is_good_sell_in_days cur date<%s> is later than end date<%s>."%((date),(self.end_date)))
                return False
            if td is None:  #未找到对应日期的数据, 继续找下一天，计数器不变
                continue
            else:   #对应日期有数据，则判断并确认是否返回
                #print("DEBUG: i-<%d>,  date-<%s> - %s"%(i,date, td.p.is_good_buy(bp)))
                if self.is_good_sell_in(sp, date):   #真值，直接返回
                    return True
                i += 1  #否则继续循环
        return False    #一直无真值，则直接返回失败
                
    
#非正式使用，存储一只股票/指数的所有日期的涨跌幅数据至csv文件中
#便于合理划分预测区间用
def save_pct_chg_to_csv():
    pct_chg_list = []
    s = StockData('600036.SH', asset='E')
    for td in reversed(s.trade_datas):
        try:
            #print("DEBUG: date<%s>, t1_low/high is-<%.2f/%.2f>, T2_low/high is-<%s/%s>, T3_low/high is-<%s/%s>"%(td.date, td.t1_low, td.t1_high, td.t2_low, td.t2_high, td.t3_low, td.t3_high))
            pct_chg_list.append([td.date, 
                                td.t1_low_chg_pct, td.t1_high_chg_pct, 
                                td.t2_low_chg_pct, td.t2_high_chg_pct, 
                                td.t3_low_chg_pct, td.t3_high_chg_pct])
            #print([np.argmax(td.t1_low_categorical), np.argmax(td.t1_high_categorical), \
            #                        np.argmax(td.t2_low_categorical), np.argmax(td.t2_high_categorical), \
            #                        np.argmax(td.t3_low_categorical), np.argmax(td.t3_high_categorical)])
        except:
            pass
    #print(pct_chg_list)
    pct_chg_file = pd.DataFrame(pct_chg_list,columns=['date', 't1_low', 't1_high', 't2_low', 't2_high', 't3_low', 't3_high'])
    pct_chg_file.to_csv("pct_chg.csv")



if __name__ == "__main__":
    s = StockData('600036.SH', asset='E')
    #sz = StockData('399001.SZ', asset='I')
    #sh = StockData('000001.SH', asset='I')
    #d = day_plus_minor(s.trade_datas[0].date,-16)
    #print("DEBUG: newest date is <%s>."%str(d))
    #print(s.is_good_sell_in(46.10, d))
    #print(s.is_good_buy_in_days(42.7, d, 5))
    
    #save_pct_chg_to_csv()