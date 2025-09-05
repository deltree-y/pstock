# coding=utf-8
import os
from model.model import StockModel
from datasets.stock import Stock
from abandoned.datasets.stockdatasets import StockDataSets
from predict.predict import Predict
from predict.strategy import Strategy
from abandoned.datasets.stockdata import StockData
from utils.const_def import REL_CODE_LIST, BIG_CODE_LIST

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def main():
    #需要在此输入预测代码
    stock_list = []
    stock_list.append(Stock('600036.SH', if_update=False))
    ts_code = '600036.SH'
    #需要在此输入预测模型的代码列表
    model_codes = ['t1l_20000_best', 't1h_20000_best', 't2l_20000_best', 't2h_20000_best', 't3l_20000_best', 't3h_20000_best']
    #model_codes = ['t1l_100_best', 't1h_100_best', 't2l_100_best', 't2h_100_best', 't3l_100_best', 't3h_100_best']

    sds = StockDataSets(ts_code, REL_CODE_LIST, BIG_CODE_LIST)
    data = sds.get_newest_dataset()
    base_price = sds.tc.get_data_by_date(sds.newest_date).close
    predict = Predict(ts_code, model_codes, data, base_price, sds.newest_date)
    strategy = Strategy(predict, if_debug_on=True)

    fo = open("Z://资料//stock//predict_daily.txt", "r+")
    file_contnet = fo.read()
    fo.seek(0,0)
    fo.write("*****************************************************************************************\n")
    fo.write("[%s] BEGIN ****************************************************************\n"%StockData(ts_code).end_date)
    fo.write("INFO: target(%s) in [%s]'s close value is <%s>\n"%(ts_code, sds.newest_date, base_price))
    fo.write(predict.get_predict_string())

    print("\nINFO: target(%s) in [%s]'s close value is <%s>"%(ts_code, sds.newest_date, base_price))
    print("----------------------------------------------------------------------------------------")
    predict.print_predict_string()
    print("----------------------------------------------------------------------------------------\n")
    
    print("买入预测: ")
    buy_price = strategy.get_buy_price()
    buy_strategy_str = strategy.get_buy_strategy_str()
    if buy_price != -1:
        print("INFO: %s"%(buy_strategy_str))
        print("INFO: buy  price is : [%.2f(%+.1f%%)]"%(buy_price, (buy_price-base_price)*100/base_price))
        fo.write("\n\nINFO:%s"%(buy_strategy_str))
        fo.write("\nINFO: buy  price is : [%.2f(%+.1f%%)]\n"%(buy_price, (buy_price-base_price)*100/base_price))
    else:
        print("INFO: NOT suggest to buy today!")
        fo.write("INFO: NOT suggest to buy today!\n")

    print("\n卖出预测: ")
    sell_price = strategy.get_sell_price()
    sell_strategy_str = strategy.get_sell_strategy_str()
    if sell_price != -1:
        print("INFO: %s"%(sell_strategy_str))
        print("INFO: sell price is : [%.2f(%+.1f%%)]"%(sell_price, (sell_price-base_price)*100/base_price))
        fo.write("\n\nINFO: %s"%(sell_strategy_str))
        fo.write("\nINFO: sell price is : [%.2f(%+.1f%%)]\n"%(sell_price, (sell_price-base_price)*100/base_price))
    else:
        print("INFO: NOT suggest to sell today!")
        fo.write("INFO: NOT suggest to sell today!\n")
    print("\n----------------------------------------")
    fo.write("#####################################\n")

    fo.write("[%s] END ******************************************************************\n\n"%sds.newest_date)
    fo.write("\n"+file_contnet)
    fo.close()

if __name__ == "__main__":
   main()
