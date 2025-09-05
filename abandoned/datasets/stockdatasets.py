# coding=utf-8
import numpy as np
import sys, os
from pathlib import Path

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from abandoned.datasets.stockdata import StockData
from utils.const_def import CONTINUOUS_DAYS #预测用数据天数
from utils.const_def import TEST_DATA_CNT
#print("INFO: sys.path is - <%s>"%(sys.path))

#StockDataSets用于存储多只股票、指数的训练数据，其中：
#   tc - (SocketData类型)目标股票 
#   ec - (SocketData类型)参考股票集，可以为多只股票
#   ic - (SocketData类型)参考指数集，可以为多只指数
class StockDataSets():
    def __init__(self, target_code, e_code_list, i_code_list, data_type=None, continuous_days = CONTINUOUS_DAYS):
        self.tc = StockData(target_code)    #目标股票
        self.ec = []                        #参考股票集，可以为多只股票
        self.ec_cnt = len(e_code_list)
        self.ic = []                        #参考指数集，可以为多只指数
        self.ic_cnt = len(i_code_list)

        self.data_type = data_type
        if self.data_type is None:
            self.data_type = 't1l'  #默认使用t1l数据
        if self.data_type not in ['t1l', 't1h', 't2l', 't2h', 't3l', 't3h']:
            print("ERROR: StockDataSets(): data_type must be one of ['t1l', 't1h', 't2l', 't2h', 't3l', 't3h']!")
            exit()
        
        #从csv文件中读入参考股票，参考指数的数据
        if self.ec_cnt != 0:
            for code in e_code_list:
                self.ec.append(StockData(code, asset='E'))
        if self.ic_cnt != 0:
            for code in i_code_list:
                self.ic.append(StockData(code, asset='I'))

        self.cd = continuous_days
        self.newest_date = None
        self.full_X = []
        self.full_Y = []
        self.correspond_date_with_X = []
        self.all_day_trade_data = self.build_all_data() #以目标股票为基准，获取其他参考股票、指数的数据
        self.build_datasets()
        self.full_X = np.array(self.full_X)
        self.full_Y = np.array(self.full_Y)
        #print("INFO: X shape is -<%s>, Y shape is -<%s>"%(str(self.X.shape),str(self.Y.shape)))
        self.test_X = self.full_X[0:TEST_DATA_CNT,]
        self.test_Y = self.full_Y[0:TEST_DATA_CNT,]
        self.test_X1 = self.full_X[0:int(TEST_DATA_CNT/2),]
        self.test_Y1 = self.full_Y[0:int(TEST_DATA_CNT/2),]
        self.test_X2 = self.full_X[int(TEST_DATA_CNT/2):TEST_DATA_CNT,]
        self.test_Y2 = self.full_Y[int(TEST_DATA_CNT/2):TEST_DATA_CNT,]
        self.X = self.full_X[TEST_DATA_CNT:,]
        self.Y = self.full_Y[TEST_DATA_CNT:,]

    
    def build_all_data(self):
        all_day_trade_data = []

        #以目标股票数据的日期范围为基准，检索出参考股票和指数的对应日期数据
        for index in range(self.tc.trade_data_cnt):
            one_day_trade_data = [] #一天的数据，包括给定的股票、基金
            one_day_trade_data.append(self.tc.trade_datas[index])
            cur_date = self.tc.trade_datas[index].date
            for item in self.ec:
                one_day_trade_data.append(item.get_data_by_date(cur_date))
            for item in self.ic:
                one_day_trade_data.append(item.get_data_by_date(cur_date))
            all_day_trade_data.append(one_day_trade_data)

            #print("DEBUG: cur_date: %s"%str(cur_date))
            #print("DEBUG: all_day_trade_data shape is -<%s>"%(str(np.array(all_day_trade_data).shape)))
            #os.system("pause")        
        return all_day_trade_data

    #构造所有的X,Y集合
    def build_datasets(self):
        self.newest_date = self.tc.trade_datas[0].date
        for index in range(3, self.tc.trade_data_cnt-self.cd):
            # X - 各种股票、指数在前CONTINUOUS_DAYS天的各类数据(为一个一维数组)
            # Y - 目标股票在T, T1T2T3对应的最低跌幅比例、最高涨幅比例
            if self.data_type == 't1l':
                self.full_Y.append(self.tc.trade_datas[index].t1_low_categorical)
            elif self.data_type == 't1h':
                self.full_Y.append(self.tc.trade_datas[index].t1_high_categorical)
            elif self.data_type == 't2l':
                self.full_Y.append(self.tc.trade_datas[index].t2_low_categorical)
            elif self.data_type == 't2h':
                self.full_Y.append(self.tc.trade_datas[index].t2_high_categorical)
            elif self.data_type == 't3l':
                self.full_Y.append(self.tc.trade_datas[index].t3_low_categorical)
            elif self.data_type == 't3h':
                self.full_Y.append(self.tc.trade_datas[index].t3_high_categorical)
            else:
                print("ERROR: StockDataSets():build_datasets() - data type error!")
                exit()
            #self.full_Y.append(self.tc.trade_datas[index].t1_low_categorical)
            #self.full_Y.append([self.tc.trade_datas[index].t1_low_categorical, self.tc.trade_datas[index].t1_high_categorical, \
            #                    self.tc.trade_datas[index].t2_low_categorical, self.tc.trade_datas[index].t2_high_categorical, \
            #                    self.tc.trade_datas[index].t3_low_categorical, self.tc.trade_datas[index].t3_high_categorical])
            #print("DEBUG: index is <%s>, date<%s>"%(str(index),self.tc.trade_datas[index].date))
            #print("DEBUG: self.tc.trade_datas[index].t1_low_categorical/t1_low is <%s>/<%.3f>"%(str(self.tc.trade_datas[index].t1_low_categorical),self.tc.trade_datas[index].t1_low))
            #print("DEBUG: Y is -<%s>"%(self.full_Y[index-3]))
            #os.system("pause")

            self.full_X.append(self.__get_dataset(self.all_day_trade_data[index:index + self.cd]))
            self.correspond_date_with_X.append(self.tc.trade_datas[index].date)
            #os.system("pause")
    
    def get_newest_dataset(self):
        return self.__get_dataset(self.all_day_trade_data[0:self.cd])
   
    #作用：将所有数据组装为一个一维数组
    def __get_dataset(self,sd_pc):
        trade_datas_for_days = np.array(sd_pc)
        trade_datas_for_days = np.transpose(trade_datas_for_days)
        #print("DEBUG: sd_pc shape is -<%s>"%(str(trade_datas_for_days.shape)))
        #os.system("pause")
        ret = None
        stock_data_index = 0
        for one_stock_data in trade_datas_for_days:  #每个item为一天的数据,类型 - TradeData()列表,包含目标、参考股票、参考指数一天的数据
            #print("DEBUG: one_day_stockdata shape is -<%s>"%(str(np.array(one_stock_data).shape)))
            #print("DEBUG:%s"%one_stock_data)
            if stock_data_index == 0:   #目标股票
                asset = 'T'
                stock_data_index = 1
            elif one_stock_data[0].asset == 'E':    #目标参考股票
                asset = 'E'
            elif one_stock_data[0].asset == 'I':    #参考指数
                asset = 'I'
            else:
                print("ERROR: StockDataSets():__get_dataset() - data type error!")
                exit()
            
            #拼接返回数据
            if ret is None:
                ret = self.__get_one_stock_data_for_days(one_stock_data, asset)
                ret = np.expand_dims(ret, axis = 2)
            else:
                ret = np.insert(ret, 0, values=np.array(self.__get_one_stock_data_for_days(one_stock_data, asset)), axis=2)
        #print("DEBUG: unit shape is -<%s>"%(str(np.array(ret).shape)))
        return ret

    #作用：根据目标股表、参考股票、参考指数这3种分类，来分别获取不同类型的的数据(每调用只获取一只股票/指数的多日数据)
    #      比如目标股票应获取最详细数据，参考股票获取数据较少
    #返回：一只股票/指数的多日数据(Tn为一个维度)
    def __get_one_stock_data_for_days(self, one_stock_data, asset):
        ret = None
        for trade_data in one_stock_data:
            if asset == 'T':
                data = self.__get_data_from_stockdata_detail(trade_data)
            elif asset == 'E':
                data = self.__get_data_from_stockdata_simple(trade_data)
            elif asset == 'I':
                data = self.__get_data_from_stockdata_index(trade_data)
            else:
                print("ERROR: StockDataSets():__get_one_stock_data_for_days() - data type error!")
                exit()  

            if ret is None:
                ret = np.array(data)
                ret = np.expand_dims(ret, axis = 1)
            else:
                ret = np.insert(ret, 0, values=np.array(data), axis=1)

        #print("DEBUG: ret.shape is -<%s>"%str(ret.shape))
        #print("DEBUG: ret is -<%s>"%(ret))
        #os.system("pause")
        return ret


    def __get_data_from_stockdata_detail(self, osd):
        if osd is not None:
            return [osd.open, osd.high, osd.low, osd.close, osd.p_close, osd.pct_chg, osd.pe, osd.pb, osd.ps,
                    osd.turnover_rate, osd.total_share, osd.float_share,
                    osd.buy_sm_vol, osd.sell_sm_vol, osd.buy_md_vol, osd.sell_md_vol, 
                    osd.buy_lg_vol, osd.sell_lg_vol, osd.buy_elg_vol, osd.sell_elg_vol, 
                    osd.net_mf_vol]
            #return [osd.open, osd.close, osd.vol, osd.buy_elg_vol, osd.sell_elg_vol]
        else:
            return None

    def __get_data_from_stockdata_simple(self, osd):
        if osd is not None:
            #return [osd.close, osd.vol]
            return [osd.close, osd.high, osd.low, osd.p_close, osd.vol, osd.buy_elg_vol, osd.sell_elg_vol,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                    #0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            #return [osd.open, osd.high, osd.low, osd.close, osd.vol, 
            #    osd.buy_sm_vol, osd.sell_sm_vol, osd.buy_elg_vol, osd.sell_elg_vol]
        else:
            return None

    def __get_data_from_stockdata_index(self, osd):
        if osd is not None:
            #return [osd.open, osd.high, osd.low, osd.close, osd.vol]
            #return [osd.open, osd.high, osd.low, osd.close]
            return [osd.open, osd.high, osd.low, osd.close, osd.p_close, osd.vol,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                    #0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            return None


if __name__ == "__main__":

    sds = StockDataSets('600036.SH',[],['399001.SZ', '000001.SH'])
    #print("X shape is :%s" % str(sds.X.shape))
    #print("Y shape is :%s" % str(sds.Y.shape))
    pass
