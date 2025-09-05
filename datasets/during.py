#-*- coding:UTF-8 -*-
from pathlib import Path
import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from stock import Stock
from abandoned.datasets.stockdata import StockData
from abandoned.datasets.stockdatasets import StockDataSets
from utils.utils import day_plus_minor

class During():
    def __init__(self, start_date, during=None, end_date=None, desc=None):
        self.start_date = start_date
        self.during = during
        self.end_date = end_date
        self.desc = desc

        self.start_date_index = -1
        self.during_date = -1
        self.date_list = []

    #根据输入的日期列表，返回目标日期所处的index序号
    def get_start_date_index(self, date_list):
        start_date = self.start_date
        self.date_list = date_list
        date_cnt = len(date_list)
        for err_cyc in range(300):  #容错300轮
            for i in range(date_cnt):
                if str(date_list[i]) == start_date:
                    self.start_date_index = i
                    self.during = self.get_during()
                    return i
            start_date = day_plus_minor(start_date,1)
        print("ERROR: During().get_start_date_index() - 对应的日期不存在于给定的日期列表中。")
        sys.exit()

    #根据输入的日期列表，和开始日期，计算during是否会超出日期列表
    def get_during(self):
        #print("DEBUG: before self.during is <%d>"%self.during)
        if self.start_date_index == -1: #若未计算过start_date_index，则直接给出构造取值
            return self.during
        elif len(self.date_list)==0:
            return self.during
        else:   #需要重新计算during
            during_cnt = 0
            date_cnt = len(self.date_list)
            for i in range(self.start_date_index, 0, -1):
                during_cnt += 1
            if during_cnt < self.during:
                self.during = during_cnt
        #print("DEBUG: after  self.during is <%d>"%self.during)
        return self.during



class StockDuring():
    def __init__(self, sds, during_list=[]):
        self.sds = sds
        self.ts_code = self.sds.tc.ts_code
        self.during_list = during_list
        self.during_count = len(during_list)
        #self.sd = StockData(self.ts_code)
        self.__build_during(self.sds.correspond_date_with_X)

    def __build_during(self, date_list):
        for during in self.during_list:
            during.get_start_date_index(date_list)
            #print("DEBUG: date[%s] index is [%d]"%(during.start_date, during.start_date_index))


if __name__ == "__main__":
    s = Stock('600036.SH', if_update=False, r=0.018, d=0.006)
    sm_list = []
    big_code_list = ['399001.SZ', '000001.SH']
    sds = StockDataSets(s.ts_code, [], big_code_list)
    
    stock_during = StockDuring(sds, [During('20200319',207,desc='上涨周期'),During('20201106',480,desc='小下降周期')])
    #stock_during = StockDuring(sds, [During('20250101',207,desc='上涨周期')])

