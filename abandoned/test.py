import os

from predict.predict import Predict
from predict.strategy import Strategy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from datetime import datetime

from abandoned.datasets.stockdatasets import StockDataSets
from abandoned.datasets.stockdata import StockData
from datasets.stock import Stock
from datasets.money import Funds
from datasets.during import StockDuring, During
from predict.cat import Cat
from utils.utils import get_string_date
from utils.const_def import REL_CODE_LIST, BIG_CODE_LIST

def main():
    #需要在此输入预测代码
    s = Stock('600036.SH', if_update=False)  #招商银行
    #s = Stock('601288.SH', if_update=False)  #农业银行
    #s = Stock('000333.SZ', if_update=False)  #美的集团

    #需要在此输入预测模型的代码列表
    model_codes = ['t1l_20000_best', 't1h_20000_best', 't2l_20000_best', 't2h_20000_best', 't3l_20000_best', 't3h_20000_best']
    FOUNDING_MONEY = 500000   #本金
    sd = StockData(s.ts_code)   #某只股票的所有交易数据
    sds = StockDataSets(s.ts_code, REL_CODE_LIST, BIG_CODE_LIST)
    #during = StockDuring(sds, [During('20240101',240,desc='24年')])
    during = StockDuring(sds, \
            [During('20150101',2400,desc='15-24年'), During('20200101',1200,desc='20-24年'), During('20210101',960,desc='21-24年'), \
             During('20220101', 720,desc='22-24年'), During('20230101', 480,desc='23-24年'), During('20240101',240,desc='24年'), \
            # During('20210210',340,desc='美的下降周期'), \
            # During('20231025',240,desc='美的上涨周期'), During('20221031',50,desc='短上涨周期'), \
             During('20201106', 480,desc='招行下降周期(2年)'), During('20211106', 240,desc='下降周期(1年)'), During('20230101',239,desc='下降周期(23年)'), \
             During('20200319', 207,desc='招行上涨周期'), During('20241126', 60,desc='短上涨周期'), \
            # During('20210316', 360,desc='农行下降周期(2年)'), \
            # During('20221104', 480,desc='农行上涨周期'), During('20221104', 120,desc='短上涨周期'), \
             During('20230925',240,desc='24年上涨前'), During('20250301',30,desc='避免过拟合周期') \
                ])
    during = StockDuring(sds, [During('20250701',240,desc='未训练数据')])

    IS_SIMPLE_PRINT = False
    IS_PRINT_BUY_SELL_INFO = True
    IF_PRINT_DETAILED_PREDICT = True
    if_debug_on = not IS_SIMPLE_PRINT

    predict = Predict(s.ts_code, model_codes)
    #循环每天的数据，并计算出针对T+n天的预测
    for dur in during.during_list:
        f = Funds(FOUNDING_MONEY)
        #t1l, t1h, t2l, t2h, t3l, t3h = 0, 0, 0, 0, 0, 0
        open_days_cnt,trade_days_cnt = 0, 0
        #print("DEBUG: start_date_index is <%d>"%start_date_index)
        for i in range(dur.start_date_index, dur.start_date_index-dur.during,-1):
            _=print("***********************************************************************************") if if_debug_on else None
            #print("DEBUG: <%s> round in <%s> start!"%(str(i),str(sds.correspond_date_with_X[i])))
            base_price = sd.get_data_by_date(sds.correspond_date_with_X[i]).close   #当前收盘价
            t = sds.correspond_date_with_X[i]    #T
            t1 = sd.get_next_trade_date(t)       #T+1
            t2 = sd.get_next_trade_date(t1)      #T+2
            t3 = sd.get_next_trade_date(t2)      #T+3
            open_days_cnt += 1
            predict.set_T(t)
            predict.set_base_price(base_price)
            predict.get_predicts(sds.full_X[i])
            strategy = Strategy(predict, if_debug_on=IF_PRINT_DETAILED_PREDICT)
            if if_debug_on:
                print("INFO: T0.close(%.2f) T1[%s] real low/high - [%.2f( %.1f%%)/%.2f( %.1f%%)]"%(base_price, t1, \
                                                                                                      sd.get_data_by_date(t1).low, 100*(sd.get_data_by_date(t1).low-base_price)/base_price, \
                                                                                                        sd.get_data_by_date(t1).high, 100*(sd.get_data_by_date(t1).high-base_price)/base_price))
            if IF_PRINT_DETAILED_PREDICT:
                predict.print_predict_string()

            if f.get_stock_quantity() == 0:
                sell_price = -1
                buy_price = strategy.get_buy_price()
            else:
                buy_price = -1
                sell_price = strategy.get_sell_price() 
            
            if not IS_SIMPLE_PRINT:
                #以下为T+1交易策略判定及执行：
                #首先进行买入判定
                if f.get_stock_quantity() > 0:
                    pass
                elif buy_price == -1:
                    print("INFO: [%s]buy_max  FAILED - strategy is IGNORE."%t1)
                elif not sd.is_good_buy_in(buy_price, t1):
                    print("INFO: [%s]buy_max  FAILED - predict[%.2f]  is WRONG!!!!!!!!!!!!!!!!!!!!!!!"%(t1,buy_price))
                else: #如果T+1可以以预测价格买到，则全买
                    buy_quantity = f.buy_max(buy_price, t1)
                    trade_days_cnt += 1
                    if buy_quantity > 0:    #如果已经成功买过，则当天不再尝试卖出交易
                        continue
                        
                #再进行卖出判定
                if f.get_stock_quantity() == 0:
                    #print("INFO: [%s]sell_all FAILED - quantity is ZERO."%t1)
                    pass
                elif sell_price == -1:
                    print("INFO: [%s]sell_all FAILED - strategy is IGNORE."%t1)
                elif not sd.is_good_sell_in(sell_price, t1):
                    print("INFO: [%s]sell_all FAILED - predict[%.2f]  is WRONG!!!!!!!!!!!!!!!!!!!!!!!"%(t1,sell_price))
                else:   #如果T+1可以以预测价格卖掉，则全卖
                    f.sell_all(sell_price, t1)
                    trade_days_cnt += 1

            if IS_SIMPLE_PRINT:
                #以下为简单判定
                if sd.is_good_buy_in(buy_price, t1) and buy_price != -1 and f.get_stock_quantity()==0: #如果T+1可以以预测价格买到，则全买
                    buy_quantity = f.buy_max(buy_price, t1, is_print=IS_PRINT_BUY_SELL_INFO)
                    trade_days_cnt += 1
                    if buy_quantity > 0:    #如果已经成功买过，则当天不再交易
                        continue
                if sd.is_good_sell_in(sell_price, t1) and f.get_stock_quantity()>0 and sell_price != -1: #如果T+1可以以预测价格卖掉，则全买
                    f.sell_all(sell_price, t1, is_print=IS_PRINT_BUY_SELL_INFO)
                    trade_days_cnt += 1

            #*****************************以下6行为统计用，不管****************************************************
            #t1l = t1l+1 if sd.is_good_buy_in_days(plg.get_t1_low_price(),t1,1) else t1l
            #t1h = t1h+1 if sd.is_good_sell_in_days(plg.get_t1_high_price(),t1,1) else t1h
            #t2l = t2l+1 if sd.is_good_buy_in_days(plg.get_t2_low_price(),t2,1) else t2l
            #t2h = t2h+1 if sd.is_good_sell_in_days(plg.get_t2_high_price(),t2,1) else t2h
            #t3l = t3l+1 if sd.is_good_buy_in_days(plg.get_t3_low_price(),t3,1) else t3l
            #t3h = t3h+1 if sd.is_good_sell_in_days(plg.get_t3_high_price(),t3,1) else t3h
            #*****************************以上6行为统计用，不管****************************************************
        
        #for 循环结束

        start_date, end_date = datetime.strptime(sds.correspond_date_with_X[dur.start_date_index],"%Y%m%d"), datetime.strptime(t1,"%Y%m%d")
        datetime_delta = end_date - start_date
        s_open, e_colse = sd.get_exact_data_by_date(get_string_date(start_date)).open, sd.get_exact_data_by_date(get_string_date(end_date)).close
        sum_profit_margin = f.get_profit_margin(sd.get_exact_data_by_date(t).close)
        yy_profit_margin = sum_profit_margin/(datetime_delta.days/365)
        last_day_amount = f.get_total_amount(sd.get_exact_data_by_date(t).close)

        print("\n***********************************************************************************************")
        print("INFO: <%s> from [%s](%.2f) to [%s](%.2f) , open_days/trade_days:[%d]/[%d]"%(dur.desc,\
                                                                                     start_date.strftime("%Y/%m/%d"), \
                                                                                 s_open, end_date.strftime("%Y/%m/%d"),\
                                                                                 e_colse, \
                                                                                 open_days_cnt,trade_days_cnt))
        #print("INFO: T1,T2,T3  buy/sell good raito is [%.2f%%/%.2f%%],[%.2f%%/%.2f%%],[%.2f%%/%.2f%%]"%(\
        #                                                                    100*t1l/(dur.during+1),100*t1h/(dur.during+1),\
        #                                                                    100*t2l/(dur.during+1),100*t2h/(dur.during+1),\
        #                                                                    100*t3l/(dur.during+1),100*t3h/(dur.during+1)))
        print("INFO: current stock quantity: <%s>, capital/current amount is <￥ %s>/<￥ %s>"%(format(int(f.get_stock_quantity()),","),\
                                                                                             format(int(f.capital),","),\
                                                                                                format(int(last_day_amount),",")))
        print("INFO: stock change:<%.0f%%>, profit margin is <sum>/<yy> : <%.0f%%>/<%.1f%%>"%(100*(e_colse-s_open)/s_open, \
                                                                                              sum_profit_margin, yy_profit_margin))
        print("***********************************************************************************************\n")


def test():
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #p = Predict()
    #print("DEBUG: %s"%str(Cat(pct_chg=2).get_cat()))
    #print("DEBUG: %s"%str(Cat(cat=[0,2,34,56,4]).classes))
    #print("DEBUG: %s"%str(Cat(classes=5).classes))
    Cat(pct_chg=2).get_cat()
    Cat(cat=[0,2,34,56,4])
    Cat(classes=5)

if __name__ == "__main__":
    main()
    #test()
