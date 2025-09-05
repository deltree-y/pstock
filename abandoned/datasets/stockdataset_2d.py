#-*- coding:UTF-8 -*-
import sys,os,time
import numpy as np
from pathlib import Path
from keras.utils import to_categorical
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from stock import Stocks
from stockinfo import StockInfo
from trade import Trade
from utils.const_def import TOKEN, LATEST_DATE, CONTINUOUS_DAYS, PREDICT_CNT, TEST_DATA_CNT

class StockDatasets():
    def __init__(self, stocks):
        self.stocks = stocks
        self.si = stocks.si
        self.stockdatasets = []
        self.y = []
        self.stocks_count = len(self.stocks.stock_list)
        for stock in self.stocks.stock_list:
            self.stockdatasets.append(StockDataset(stock))
        self.date_list = self.stockdatasets[0].trade.trade_date_list if self.stockdatasets else []
        #self.datasets_1d = np.array(self.get_1d_datasets()).transpose(0,2,1)
        self.datasets_2d = np.array(self.get_2d_datasets(),dtype=np.float32).transpose(1,3,2,0)  #训练分段,属性,天,股票
        self.datasets_2d = np.insert(self.datasets_2d, 0, np.ones_like(self.datasets_2d[:,:,:,0])*-1, axis=3)  #在前面插入一片全-1的三维数据
        self.update_y() #update self.y
        self.train_x, self.train_y = self.datasets_2d[TEST_DATA_CNT:,:,:,:], self.y[TEST_DATA_CNT:,:,:]
        self.test_x, self.test_y = self.datasets_2d[:TEST_DATA_CNT,:,:,:], self.y[:TEST_DATA_CNT,:,:]

        #print("INFO: self.datasets_2d.shape:%s, self.y.shape:%s"%(str(self.datasets_2d.shape),str(self.y.shape)))
        print("INFO: self.train_x.shape:%s, self.train_y.shape:%s"%(str(self.train_x.shape),str(self.train_y.shape)))
        print("INFO: self.test_x.shape:%s, self.test_y.shape:%s"%(str(self.test_x.shape),str(self.test_y.shape)))
        print("INFO: latest date is <%s>"%self.stocks.stock_list[0].latest_date)
    
    def update_y(self):    #注意若未找到符合匹配要求的数据,则序号以0表示
        print("INFO: start to build y...")
        self.y.append(self.list_to_categorical([0]*PREDICT_CNT))  #第一组x数据为全-1,对应的y为全0
        no_match_cnt, one_match_cnt, full_match_cnt = 0, 0, 0
        for i in range(len(self.date_list[:self.stockdatasets[0].data_count-1])):
            match_ts_code_list = []
            for stockdataset in self.stockdatasets:
                if stockdataset.expect_mark[i]:
                    match_ts_code_list.append(stockdataset.stock.ts_code)
            if len(match_ts_code_list) > 0:   #
                top_n_ts_code_list = self.get_top_n_y(match_ts_code_list)
                one_match_cnt = one_match_cnt+1
                full_match_cnt = full_match_cnt+1 if len(top_n_ts_code_list)>=PREDICT_CNT else full_match_cnt
                top_n_no_list = self.get_no_list_by_ts_code_list(top_n_ts_code_list)
                top_n_no_list = top_n_no_list + [0]*(PREDICT_CNT-len(top_n_no_list))  #若不够足量要求,则以0补齐
            else:
                top_n_no_list = [0]*PREDICT_CNT
                no_match_cnt += 1
            self.y.append(self.list_to_categorical(top_n_no_list))
        print("INFO: no_match/one_match/full_match cnt<%d(%.2f%%)>/<%d(%.2f%%)>/<%d(%.2f%%)>" % \
              (no_match_cnt, no_match_cnt / (i + 1) * 100, one_match_cnt, one_match_cnt / (i + 1) * 100, full_match_cnt, full_match_cnt / (i + 1) * 100))
        self.y = np.array(self.y)
    
    def list_to_categorical(self, list):
        categorical_list = []
        for k in list:
            categorical_list.append(to_categorical(k,num_classes=self.stocks_count+1))
        return categorical_list

    
    def get_1d_datasets(self):
        return [dataset.get_1d_datasets() for dataset in self.stockdatasets]
    
    def get_2d_datasets(self):
        last_shape = None
        data_list = []
        for dataset in self.stockdatasets:
            ret = dataset.data_2d
            data_list.append(ret)
            if last_shape is not None:
                if last_shape != np.array(ret).shape:
                    print("ERROR: get_2d_datasets() - <%s> shape mismatch. last_shape:%s, current_shape:%s" % (dataset.stock.name, last_shape, np.array(ret).shape))
                    exit()
            last_shape = np.array(ret).shape

        print("Info: get_2d_datasets() done.")
        #TODO: 需要在0位置插入一个全-1的记录
        return data_list
    
    #从给出的股票代码列表中,选出市值最高的前N只股票代码
    def get_top_n_y(self, ts_code_list):
        ret = self.stocks.get_sort_by_mv(ts_code_list)
        if len(ret) > PREDICT_CNT:  #最多返回PREDICT_CNT个数据
            ret = ret[:PREDICT_CNT]
        return ret

    #从给出的股票代码列表中,返回对应的datasets序号列表
    def get_no_list_by_ts_code_list(self, ts_code_list):
        no_list = []
        for ts_code in ts_code_list:
            for i, dataset in enumerate(self.stockdatasets, start=1):
                if dataset.stock.ts_code == ts_code:
                    no_list.append(i)
        return no_list

    #根据datasets序号返回对应的股票代码
    def get_ts_code_by_no(self, no):
        for i, dataset in enumerate(self.stockdatasets, start=1):
            if i == no:
                return dataset.stock.ts_code
        return None

    def print_fit_data(self):
        for date in self.date_list:
            print("INFO: <%s> -" % (date))
            for dataset in self.stockdatasets:
                if len(dataset.trade.fit_datas)>0:
                    if date in dataset.trade.fit_datas[:,0]:
                        print("INFO:             <%s(%s)>, t1/t2:<%+.3f%%/%+.3f%%>, profit:<%.2f%%>"\
                               % (dataset.stock.name,dataset.stock.ts_code,\
                                    dataset.trade.fit_datas[dataset.trade.fit_datas[:,0]==date, -2]*100, \
                                    dataset.trade.fit_datas[dataset.trade.fit_datas[:,0]==date, -1]*100, \
                                    (dataset.trade.fit_datas[dataset.trade.fit_datas[:,0]==date, -1]-dataset.trade.fit_datas[dataset.trade.fit_datas[:,0]==date, -2])*100))


class StockDataset():
    def __init__(self, stock, lines=CONTINUOUS_DAYS, is_empty=False):
        self.stock = stock
        self.lines = lines
        self.trade = Trade(stock)
        self.data_2d = []
        self.update_2d_datasets()   #更新self.data_2d
        self.data_count = self.data_2d.shape[0] if self.data_2d is not None else 0
        self.expect_mark = self.get_expect_mark()
        #self.list = self.get_1d_datasets()
        #self.data_count = self.list.shape[0] if self.list is not None else 0
        #print(self.matrix.shape)
        #print("DEBUG: Stock <%s(%s)> - Data Count: %d" % (self.stock.name, self.stock.ts_code, self.data_count))

        #for line in self.list:
        #    print("DEBUG: data<%s>, fit_mark<%s>" % (line[:5], line[-1]))

    def get_expect_mark(self):
        if self.data_2d is None or len(self.data_2d) == 0:
            print("ERROR: get_expect_mark() - no refer data. will exit.")
            exit()
        return self.trade.fit_mark[:self.data_count-1]

    #返回该股票的所有数据，除最后一列为结果(0/1)外，其余全部为可训练数据，第二维度为训练数据分片
    def get_1d_datasets(self):
        list = []
        for i in range(len(self.trade.__trade_datas) - self.lines + 1):
            temp_matrix = self.trade.__trade_datas[i:i + self.lines]
            one_line = []
            for line in temp_matrix:
                one_line.extend(line)
            one_line.extend([self.trade.fit_mark[i]])
            list.append(one_line)
        return np.array(list)

    #返回该股票的所有数据（第三维度为训练数据分片）
    #格式: [训练分段, 属性, 包含n天的一次数据]
    def update_2d_datasets(self):
        for_cnt = len(self.trade.__trade_datas) - self.lines + 1
        for i in range(for_cnt):
            self.data_2d.append((self.trade.__trade_datas[i:i + self.lines]))
        self.data_2d = np.array(self.data_2d)#.transpose(0,2,1)
        #self.data_2d = np.insert(self.data_2d, 0, self.get_2d_empty_datasets(), axis=0)  #在最前面插入一片全-1的二维数据
        #print("DEBUG: processing - %s(%s). shape<%s>"%(self.stock.name, self.stock.ts_code, matrix.shape))
    
    #返回一片全为-1的二维训练数据分片
    def get_2d_empty_datasets(self):
        if self.data_2d is None or len(self.data_2d) == 0:
            print("ERROR: get_empty_datasets() - no refer data. will exit.")
            exit()
        #return np.ones_like(self.matrix[:,:,0])[:,:-1]*-1
        return np.ones_like(self.data_2d[0,:,:])*-1

if __name__ == "__main__":
    si = StockInfo(TOKEN)
    download_list = si.get_filtered_stock_list(mmv=3000000)
    #download_list = ['603993.SH']
    sse = Stocks(download_list, si, start_date=str(LATEST_DATE), end_date='20250825', if_force_download=False)
    sd = StockDatasets(sse)