# coding=utf-8
import sys, os
from pathlib import Path
from utils.const_def import BIG_INCREASE_THRESHOLD, BIG_DROP_THRESHOLD
from utils.utils import SuperList, get_mind_value
from collections import defaultdict

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

#用于处理预测结果，基于输入的T1,T2,T3 low/high预测值，生成对应的买卖价格
#其中predict参数为Predict类的实例
#predict参数包含了T1,T2,T3 low/high预测值
class Strategy:
    def __init__(self, predict, if_debug_on=False):
        self.predict = predict
        self.if_debug_on = if_debug_on
        self.buy_strategy_str = None
        self.sell_strategy_str = None
        self.buy_price = -1
        self.sell_price = -1
        self.predict_result = None
        self.Strategy = defaultdict(SuperList)  # 用于存储基于高价的策略
        self.__build_strategy__()

    #*******************************************************************************************
    #   is_t1h_higher_than_t0()用于判断T1 low预测低值是否高于T0
    #   返回值为布尔值, True表示T1 low预测低值高于T0, False表示不高于
    #   注意：T1 low预测低值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t1h_higher_than_t0(self):
        t1h_tv = self.predict.t1h_predict.average_tv
        return t1h_tv > 0

    #*******************************************************************************************
    #   is_t2h_higher_than_t1()用于判断T2 high预测高值是否高于T1 high预测高值
    #   返回值为布尔值, True表示T2 high预测高值高于T1 high预测高值, False表示不高于
    #   注意：T2 high预测高值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t2h_higher_than_t1(self):
        t2h_tv = self.predict.t2h_predict.average_tv
        t1h_tv = self.predict.t1h_predict.average_tv
        return t2h_tv > t1h_tv

    #*******************************************************************************************
    #   is_t2h_higher_than_t0()用于判断T2 high预测高值是否高于T0
    #   返回值为布尔值, True表示T2 high预测高值高于T0, False表示不高于
    #   注意：T2 high预测高值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t2h_higher_than_t0(self):
        t2h_tv = self.predict.t2h_predict.average_tv
        return t2h_tv > 0
    
    #*******************************************************************************************
    #   is_t3h_higher_than_t2()用于判断T3 high预测高值是否高于T2 high预测高值
    #   返回值为布尔值, True表示T3 high预测高值高于T2 high预测高值, False表示不高于
    #   注意：T3 high预测高值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t3h_higher_than_t2(self):
        t3h_tv = self.predict.t3h_predict.average_tv
        t2h_tv = self.predict.t2h_predict.average_tv
        return t3h_tv > t2h_tv
    
    #*******************************************************************************************
    #   is_t3h_higher_than_t1()用于判断T3 high预测高值是否高于T1 high预测高值
    #   返回值为布尔值, True表示T3 high预测高值高于T1 high预测高值, False表示不高于
    #   注意：T3 high预测高值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t3h_higher_than_t1(self):
        t3h_tv = self.predict.t3h_predict.average_tv
        t1h_tv = self.predict.t1h_predict.average_tv
        return t3h_tv > t1h_tv
    
    #*******************************************************************************************
    #   is_t3h_higher_than_t0()用于判断T3 high预测高值是否高于T0
    #   返回值为布尔值, True表示T3 high预测高值高于T0, False表示不高于
    #   注意：T3 high预测高值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t3h_higher_than_t0(self):
        t3h_tv = self.predict.t3h_predict.average_tv
        return t3h_tv > 0

    #*******************************************************************************************
    #   is_t1l_lower_than_t0()用于判断T1 low预测低值是否低于T0
    #   返回值为布尔值, True表示T1 low预测低值低于T0, False表示不低于
    #   注意：T1 low预测低值为PredictProc类的average_tv属性 
    #*******************************************************************************************
    def is_t1l_lower_than_t0(self):
        t1l_tv = self.predict.t1l_predict.average_tv
        return t1l_tv < 0

    #*******************************************************************************************
    #   is_t2l_lower_than_t1()用于判断T2 low预测低值是否低于T1 low预测低值
    #   返回值为布尔值, True表示T2 low预测低值低于T1 low预测低值, False表示不低于
    #   注意：T2 low预测低值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t2l_lower_than_t1(self):
        t2l_tv = self.predict.t2l_predict.average_tv
        t1l_tv = self.predict.t1l_predict.average_tv
        return t2l_tv < t1l_tv
    
    #*******************************************************************************************
    #   is_t3l_lower_than_t2()用于判断T3 low预测低值是否低于T2 low预测低值
    #   返回值为布尔值, True表示T3 low预测低值低于T2 low预测低值, False表示不低于
    #   注意：T3 low预测低值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t3l_lower_than_t2(self):
        t3l_tv = self.predict.t3l_predict.average_tv
        t2l_tv = self.predict.t2l_predict.average_tv
        return t3l_tv < t2l_tv

    #*******************************************************************************************
    #   is_t1h_big_increase()用于判断T1 high预测低值是否大幅上涨
    #   返回值为布尔值, True表示T1 high预测低值大幅上涨, False表示不大幅上涨
    #   注意：T1 high预测低值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def is_t1h_big_increase(self):
        t1h_tv = self.predict.t1h_predict.average_tv
        return t1h_tv >= BIG_INCREASE_THRESHOLD
    
    #*******************************************************************************************
    #   is_t2h_big_increase()用于判断T2 high预测值是否大幅上涨
    #   返回值为布尔值, True表示T2 high预测值大幅上涨, False表示不大幅上涨
    #   注意：T2 high预测值为PredictProc类的average_tv属性  
    #*******************************************************************************************
    def is_t2h_big_increase(self):
        t2h_tv = self.predict.t2h_predict.average_tv
        return t2h_tv >= BIG_INCREASE_THRESHOLD
    
    #*******************************************************************************************
    #   is_t3h_big_increase()用于判断T3 high预测值是否大幅上涨
    #   返回值为布尔值, True表示T3 high预测值大幅上涨, False表示不大幅上涨
    #   注意：T3 high预测值为PredictProc类的average_tv属性  
    #*******************************************************************************************
    def is_t3h_big_increase(self):
        t3h_tv = self.predict.t3h_predict.average_tv
        return t3h_tv >= BIG_INCREASE_THRESHOLD
    
    #*******************************************************************************************
    #   get_lowest_day()用于获取T1,T2,T3 low预测值中最低的那一天
    #   返回值为字符串, '1'表示T1 low预测值最低, '2'表示T2 low预测值最低, '3'表示T3 low预测值最低, '0'表示T0最低
    #   注意：T1,T2,T3 low预测值为PredictProc类的average_tv属性
    #*******************************************************************************************
    def get_lowest_day(self):
        min_value =  min(self.predict.t1l_predict.average_tv, self.predict.t2l_predict.average_tv, self.predict.t3l_predict.average_tv, 0)
        if min_value == self.predict.t1l_predict.average_tv:
            return 'l1'
        elif min_value == self.predict.t2l_predict.average_tv:
            return 'l2'
        elif min_value == self.predict.t3l_predict.average_tv:
            return 'l3'
        return 'l0'

    #*******************************************************************************************
    #   根据T1,T2,T3 low/high预测值, 生成买入/卖出价格
    #   返回值为买入/卖出价格, 如果不买/不卖则返回-1
    #*******************************************************************************************
    def ___sort_predict_price(self):
        if self.is_t1h_higher_than_t0():#t1>t0
            if self.is_t2h_higher_than_t1():#t2>t1>t0
                if self.is_t3h_higher_than_t2():#t3>t2>t1>t0:
                    self.predict_result = "h3210"
                else:#t2>t1>t0, t2>t3
                    if self.is_t3h_higher_than_t1():#t2>t3>t1>t0
                        self.predict_result = "h2310"
                    else:#t2>t1>t0, t2>t1>t3
                        if self.is_t3h_higher_than_t0():#t2>t1>t3>t0
                            self.predict_result = "h2130"
                        else:#t2>t1>t0>t3
                            self.predict_result = "h2103"
            else:#t1>t0, t1>t2
                if self.is_t2h_higher_than_t0():#t1>t2>t0
                    if self.is_t3h_higher_than_t2():#t1>t2>t0, t3>t2
                        if self.is_t3h_higher_than_t1():#t3>t1>t2>t0
                            self.predict_result = "h3120"
                        else:#t1>t3>t2>t0
                            self.predict_result = "h1320"
                    else:#t1>t2>t0, t2>t3
                            if self.is_t3h_higher_than_t0():#t1>t2>t3>t0
                                self.predict_result = "h1230"
                            else:#t1>t2>t0>t3
                                self.predict_result = "h1203"
                else:#t1>t0>t2, t0>t2
                    if self.is_t3h_higher_than_t1():#t3>t1>t0>t2, t3>t1
                        self.predict_result = "h3102"
                    else:#t1>t0>t2, t1>t3
                        if self.is_t3h_higher_than_t0():#t1>t3>t0>t2, t3>t0
                            self.predict_result = "h1302"
                        else:#t1>t0>t2, t0>t3
                            if self.is_t3h_higher_than_t2():#t1>t0>t3>t2, t3>t2
                                self.predict_result = "h1032"
                            else:#t1>t0>t2>t3, t2>t3
                                self.predict_result = "h1023"
        
        else: #t0>t1
            if self.is_t2h_higher_than_t0():#t2>t0>t1, t2>t0
                if self.is_t3h_higher_than_t2():#t3>t2>t0>t1, t3>t2
                    self.predict_result = "h3201"
                else:#t2>t0>t1, t2>t3
                    if self.is_t3h_higher_than_t0():#t2>t3>t0>t1, t3>t0
                        self.predict_result = "h2301"
                    else:#t2>t0>t1, t0>t3
                        if self.is_t3h_higher_than_t1():#t2>t0>t3>t1, t3>t1
                            self.predict_result = "h2031"
                        else:#t2>t0>t1>t3, t1>t3
                            self.predict_result = "h2013"
            else:#t0>t1, t0>t2, b.b
                if self.is_t2h_higher_than_t1():#t0>t2>t1, t2>t1, b.b.a
                    if self.is_t3h_higher_than_t0():#t3>t0>t2>t1, t3>t0
                        self.predict_result = "h3021"
                    else:#t0>t2>t1, t0>t3
                        if self.is_t3h_higher_than_t2():#t0>t3>t2>t1, t3>t2
                            self.predict_result = "h0321"
                        else:#t0>t2>t1, t0>t3, t2>t3
                            if self.is_t3h_higher_than_t1():#t0>t2>t3>t1, t3>t1
                                self.predict_result = "h0231"
                            else:#t0>t2>t1>t3, t1>t3
                                self.predict_result = "h0213"
                else:#t0>t1>t2, t1>t2, b.b.b
                    if self.is_t3h_higher_than_t0():#t3>t0>t1>t2, t3>t0
                        self.predict_result = "h3012"
                    else:#t0>t1>t2, t0>t3
                        if self.is_t3h_higher_than_t1():#t0>t3>t1>t2, t3>t1
                            self.predict_result = "h0312"
                        else:#t0>t1>t2, t0>t3, t1>t3
                            if self.is_t3h_higher_than_t2():#t0>t1>t3>t2, t3>t2
                                self.predict_result = "h0132"
                            else:#t0>t1>t2>t3, t2>t3
                                self.predict_result = "h0123"
        min_low_day_str = self.get_lowest_day()
        self.predict_result += min_low_day_str
        return self.predict_result

    def __build_strategy__(self):
        #根据T1,T2,T3 low/high预测值的排序结果, 生成买入/卖出策略
        #注意：这里的排序结果是基于T1,T2,T3 low/high预测值的预测值的高低进行排序的, 而不是实际的价格
        self.Strategy['h0123'].append("INFO: (T0>t1>t2>t3)买入策略: 不买").append('-1')\
                              .append("INFO: (T0>t1>t2>t3)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h0132'].append("INFO: (T0>t1>t3>t2)买入策略: 不买").append('-1')\
                              .append("INFO: (T0>t1>t3>t2)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h0213'].append("INFO: (T0>t2>t1>t3)买入策略: 不买").append('-1')\
                              .append("INFO: (T0>t2>t1>t3)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h0231'].append("INFO: (T0>t2>t3>t1)买入策略: 不买").append('-1')\
                              .append("INFO: (T0>t2>t3>t1)卖出策略: 快卖,卖好 - 以T2h的低值卖出").append('t2h_l')
        self.Strategy['h0231l1'].append("INFO: (T0>t2>t3>t1,t1lst)买入策略: 买低 - 以t1l的低值买入").append('t1l_l')\
                                .append("INFO: (T0>t2>t3>t1,t1lst)卖出策略: 快卖,卖好 - 以T2h的低值卖出").append('t2h_l')

        self.Strategy['h0312'].append("INFO: (T0>t3>t1>t2)买入策略: 不买").append('-1')\
                              .append("INFO: (T0>t3>t1>t2)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h0312l2'].append("INFO: (T0>t3>t1>t2,t2lst)买入策略: 买低 - 以t2l的低值买入").append('t2l_l')\
                                .append("INFO: (T0>t3>t1>t2,t2lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')

        self.Strategy['h0321'].append("INFO: (T0>t3>t2>t1)买入策略: 不买").append('-1')\
                              .append("INFO: (T0>t3>t2>t1)卖出策略: 快卖,卖好 - 以T2h的低值卖出").append('t2h_l')
        self.Strategy['h0321l1'].append("INFO: (T0>t3>t2>t1,t1lst)买入策略: 买低 - 以t1l的低值买入").append('t1l_l')\
                                .append("INFO: (T0>t3>t2>t1,t1lst)卖出策略: 快卖,卖好 - 以T2h的低值卖出").append('t2h_l')

        self.Strategy['h1023'].append("INFO: (t1>T0>t2>t3)买入策略: 不买").append('-1')\
                              .append("INFO: (t1>T0>t2>t3)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h1023l3'].append("INFO: (t1>T0>t2>t3,t3lst)买入策略: 买低 - 以t3l的低值买入").append('t3l_l')\
                                .append("INFO: (t1>T0>t2>t3,t3lst)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')

        self.Strategy['h1032'].append("INFO: (t1>T0>t3>t2)买入策略: 不买").append('-1')\
                              .append("INFO: (t1>T0>t3>t2)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h1032l2'].append("INFO: (t1>T0>t3>t2,t2lst)买入策略: 以t2l的低值买入").append('t2l_l')\
                                .append("INFO: (t1>T0>t3>t2,t2lst)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        
        self.Strategy['h1203'].append("INFO: (t1>t2>T0>t3)买入策略: 买低 - 以T1l,T2l的低值低者买入").append('t1l_l')\
                              .append("INFO: (t1>t2>T0>t3)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h1203l1'].append("INFO: (t1>t2>T0>t3,t1lst)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t1>t2>T0>t3,t1lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1203l2'].append("INFO: (t1>t2>T0>t3,t2lst)买入策略: 买低 - 以T2l的低值买入").append('t2l_l')\
                                .append("INFO: (t1>t2>T0>t3,t2lst)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')

        self.Strategy['h1230'].append("INFO: (t1>t2>t3>T0)买入策略: 能买就买 - 则以T1l的中值买入").append('t1l_m')\
                              .append("INFO: (t1>t2>t3>T0)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1230l1'].append("INFO: (t1>t2>t3>T0,t1lst)买入策略: 快买 - 则以T1l的高值买入").append('t1l_h')\
                                .append("INFO: (t1>t2>t3>T0,t1lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1230l2'].append("INFO: (t1>t2>t3>T0,t2lst)买入策略: 买好 - 则以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t1>t2>t3>T0,t2lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        
        self.Strategy['h1302'].append("INFO: (t1>t3>T0>t2)买入策略: 买低 - 以T1l,T2l的低值低者买入").append('t1l_l')\
                              .append("INFO: (t1>t3>T0>t2)卖出策略: 不急卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1302l1'].append("INFO: (t1>t3>T0>t2,t1lst)买入策略: 买低 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t1>t3>T0>t2,t1lst)卖出策略: 不急卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1302l2'].append("INFO: (t1>t3>T0>t2,t2lst)买入策略: 买低 - 以T2l的中值买入").append('t2l_m')\
                                .append("INFO: (t1>t3>T0>t2,t2lst)卖出策略: 卖好 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1302l3'].append("INFO: (t1>t3>T0>t2,t3lst)买入策略: 买低 - 以T3l的高值买入").append('t3l_h')\
                                .append("INFO: (t1>t3>T0>t2,t3lst)卖出策略: 卖好 - 以T1h的中值卖出").append('t1h_m')

        self.Strategy['h1320'].append("INFO: (t1>t3>t2>T0)买入策略: 买低 - 以T1l,T2l的中值低者买入").append('t1l_m')\
                              .append("INFO: (t1>t3>t2>T0)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1320l1'].append("INFO: (t1>t3>t2>T0,t1lst)买入策略: 买低 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t1>t3>t2>T0,t1lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h1320l2'].append("INFO: (t1>t3>t2>T0,t2lst)买入策略: 买低 - 以T2l的中值买入").append('t2l_m')\
                                .append("INFO: (t1>t3>t2>T0,t2lst)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h1320l3'].append("INFO: (t1>t3>t2>T0,t3lst)买入策略: 买低 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t1>t3>t2>T0,t3lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')

        self.Strategy['h2013'].append("INFO: (t2>T0>t1>t3)买入策略: 买低 - 以T1l,T2l,T3l的低值低者买入").append('t1l_l')\
                              .append("INFO: (t2>T0>t1>t3)卖出策略: 不急卖 - 以T2h的低值卖出").append('t2h_l')
        self.Strategy['h2013l1'].append("INFO: (t2>T0>t1>t3,t1lst)买入策略: 买低 - 以t1l的低值买入").append('t1l_l')\
                                .append("INFO: (t2>T0>t1>t3,t1lst)卖出策略: 不急卖 - 以T2h的低值卖出").append('t2h_l')
        self.Strategy['h2013l2'].append("INFO: (t2>T0>t1>t3,t2lst)买入策略: 买低 - 以t1l的低值买入").append('t1l_l')\
                                .append("INFO: (t2>T0>t1>t3,t2lst)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2013l3'].append("INFO: (t2>T0>t1>t3,t3lst)买入策略: 买低 - 以t3l的中值买入").append('t3l_m')\
                                .append("INFO: (t2>T0>t1>t3,t3lst)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')

        self.Strategy['h2031'].append("INFO: (t2>T0>t3>t1)买入策略: 买低 - 以T1l,T2l,T3l的中值低者买入").append('t1l_l')\
                              .append("INFO: (t2>T0>t3>t1)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2031l2'].append("INFO: (t2>T0>t3>t1,t2lst)买入策略: 不急买 - 以T2l的中值买入").append('t2l_m')\
                                .append("INFO: (t2>T0>t3>t1,t2lst)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')

        self.Strategy['h2103'].append("INFO: (t2>t1>T0>t3)买入策略: 买低 - 以T1l,T2l的中值低者买入").append('t1l_m')\
                              .append("INFO: (t2>t1>T0>t3)卖出策略: 高卖 - 以T2h的低值卖出").append('t2h_l')
        self.Strategy['h2103l1'].append("INFO: (t2>t1>T0>t3,t1lst)买入策略: 买低 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t1>T0>t3,t1lst)卖出策略: 高卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2103l2'].append("INFO: (t2>t1>T0>t3,t2lst)买入策略: 买低 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t1>T0>t3,t2lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h2103l3'].append("INFO: (t2>t1>T0>t3,t3lst)买入策略: 买低 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t1>T0>t3,t3lst)卖出策略: 卖好 - 以T2h的中值卖出").append('t2h_m')

        self.Strategy['h2130'].append("INFO: (t2>t1>t3>T0)买入策略: 好价格买 - 以T1l的低值买入").append('t1l_l')\
                              .append("INFO: (t2>t1>t3>T0)卖出策略: 高卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2130l1'].append("INFO: (t2>t1>t3>T0,t1lst)买入策略: 买好 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t1>t3>T0,t1lst)卖出策略: 高卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2130l2'].append("INFO: (t2>t1>t3>T0,t2lst)买入策略: 买好 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t1>t3>T0,t2lst)卖出策略: 卖好 - 以T2h的低值卖出").append('t2h_l')

        self.Strategy['h2301'].append("INFO: (t2>t3>T0>t1)买入策略: 买低 - 以T1l,T2l的中值低者买入").append('t1l_m')\
                              .append("INFO: (t2>t3>T0>t1)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2301l1'].append("INFO: (t2>t3>T0>t1,t1lst)买入策略: 快买 - 以T1l的高值买入").append('t1l_h')\
                                .append("INFO: (t2>t3>T0>t1,t1lst)卖出策略: 卖好 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2301l2'].append("INFO: (t2>t3>T0>t1,t2lst)买入策略: 快买 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t3>T0>t1,t2lst)卖出策略: 卖好 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h2301l3'].append("INFO: (t2>t3>T0>t1,t3lst)买入策略: 快买 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t2>t3>T0>t1,t3lst)卖出策略: 卖好 - 以T2h的中值卖出").append('t2h_m')

        self.Strategy['h2310'].append("INFO: (t2>t3>t1>T0)买入策略: 快买 - 以T1l的高值买入").append('t1l_h')\
                              .append("INFO: (t2>t3>t1>T0)卖出策略: 高卖 - 以T2h的高值卖出").append('t2h_h')

        self.Strategy['h3012'].append("INFO: (t3>T0>t1>t2)买入策略: 买低 - 以T1l,T2l的低值低者买入").append('t1l_l')\
                              .append("INFO: (t3>T0>t1>t2)卖出策略: 卖好 - 以T1h的高值卖出").append('t1h_h')
        self.Strategy['h3012l1'].append("INFO: (t3>T0>t1>t2,t1lst)买入策略: 买低 - 以T1l中值买入").append('t1l_m')\
                                .append("INFO: (t3>T0>t1>t2,t1lst)卖出策略: 卖好 - 以T1h的高值卖出").append('t1h_h')
        self.Strategy['h3012l2'].append("INFO: (t3>T0>t1>t2,t2lst)买入策略: 买低 - 以T2l中值买入").append('t2l_m')\
                                .append("INFO: (t3>T0>t1>t2,t2lst)卖出策略: 卖好 - 以T1h的高值卖出").append('t1h_h')
        self.Strategy['h3012l3'].append("INFO: (t3>T0>t1>t2,t3lst)买入策略: 买低 - 以T1l低值买入").append('t1l_l')\
                                .append("INFO: (t3>T0>t1>t2,t3lst)卖出策略: 卖好 - 以T1h的高值卖出").append('t1h_h')

        self.Strategy['h3021'].append("INFO: (t3>T0>t2>t1)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                              .append("INFO: (t3>T0>t2>t1)卖出策略: 不急卖 - 以T3h的低值卖出").append('t3h_l')
        self.Strategy['h3021l1'].append("INFO: (t3>T0>t2>t1,t1lst)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t3>T0>t2>t1,t1lst)卖出策略: 不急卖 - 以T3h的低值卖出").append('t3h_l')
        self.Strategy['h3021l2'].append("INFO: (t3>T0>t2>t1,t2lst)买入策略: 买低 - 以T2l的中值买入").append('t2l_m')\
                                .append("INFO: (t3>T0>t2>t1,t2lst)卖出策略: 不急卖 - 以T3h的低值卖出").append('t3h_l')

        self.Strategy['h3102'].append("INFO: (t3>t1>T0>t2)买入策略: 买低 - 以T1l中值买入").append('t1l_m')\
                              .append("INFO: (t3>t1>T0>t2)卖出策略: 快卖 - 以T1h的低值卖出").append('t1h_l')
        self.Strategy['h3102l1'].append("INFO: (t3>t1>T0>t2,t1lst)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t3>t1>T0>t2,t1lst)卖出策略: 卖好 - 以T3h的低值卖出").append('t3h_l')
        self.Strategy['h3102l2'].append("INFO: (t3>t1>T0>t2,t2lst)买入策略: 买低 - 以T2l的中值买入").append('t2l_m')\
                                .append("INFO: (t3>t1>T0>t2,t2lst)卖出策略: 卖好 - 以T3h的低值卖出").append('t3h_l')
        self.Strategy['h3102l3'].append("INFO: (t3>t1>T0>t2,t3lst)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t3>t1>T0>t2,t3lst)卖出策略: 卖好 - 以T3h的低值卖出").append('t3h_l')

        self.Strategy['h3120'].append("INFO: (t3>t1>t2>T0)买入策略: 快买 - 以T1l的中值买入").append('t1l_m')\
                              .append("INFO: (t3>t1>t2>T0)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h3120l1'].append("INFO: (t3>t1>t2>T0,t1lst)买入策略: 快买 - 以T1l的中值买入").append('t1l_m')\
                                .append("INFO: (t3>t1>t2>T0,t1lst)卖出策略: 卖好 - 以T3h的低值卖出").append('t3h_l')
        self.Strategy['h3120l2'].append("INFO: (t3>t1>t2>T0,t2lst)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t3>t1>t2>T0,t2lst)卖出策略: 快卖 - 以T1h的中值卖出").append('t1h_m')
        self.Strategy['h3120l3'].append("INFO: (t3>t1>t2>T0,t3lst)买入策略: 买低 - 以T1l的低值买入").append('t1l_l')\
                                .append("INFO: (t3>t1>t2>T0,t3lst)卖出策略: 卖好 - 以T3h的低值卖出").append('t3h_l')

        self.Strategy['h3201'].append("INFO: (t3>t2>T0>t1)买入策略: 买低 - 以T1l,T2l的中值低者买入").append('t1l_m')\
                              .append("INFO: (t3>t2>T0>t1)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h3201l1'].append("INFO: (t3>t2>T0>t1,t1lst)买入策略: 快买 - 以T1l高值买入").append('t1l_h')\
                                .append("INFO: (t3>t2>T0>t1,t1lst)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h3201l2'].append("INFO: (t3>t2>T0>t1,t2lst)买入策略: 不急买 - 以T1l低值买入").append('t1l_l')\
                                .append("INFO: (t3>t2>T0>t1,t2lst)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h3201l3'].append("INFO: (t3>t2>T0>t1,t3lst)买入策略: 不急买 - 以T1l中值买入").append('t1l_m')\
                                .append("INFO: (t3>t2>T0>t1,t3lst)卖出策略: 不急卖 - 以T2h的中值卖出").append('t2h_m')

        self.Strategy['h3210'].append("INFO: (t3>t2>t1>T0)买入策略: 快买 - 以T1l的高值买入").append('t1l_h')\
                              .append("INFO: (t3>t2>t1>T0)卖出策略: 高卖 - 以T2h的高值卖出").append('t2h_h')
        self.Strategy['h3210l1'].append("INFO: (t3>t2>t1>T0,t1lst)买入策略: 快买 - 以T1l的高值买入").append('t1l_h')\
                                .append("INFO: (t3>t2>t1>T0,t1lst)卖出策略: 高卖 - 以T2h的高值卖出").append('t2h_h')
        self.Strategy['h3210l2'].append("INFO: (t3>t2>t1>T0,t2lst)买入策略: 快买 - 以T1l的高值买入").append('t1l_h')\
                                .append("INFO: (t3>t2>t1>T0,t2lst)卖出策略: 稍快卖 - 以T2h的中值卖出").append('t2h_m')
        self.Strategy['h3210l3'].append("INFO: (t3>t2>t1>T0,t3lst)买入策略: 快买 - 以T1l的高值买入").append('t1l_h')\
                                .append("INFO: (t3>t2>t1>T0,t3lst)卖出策略: 稍快高卖 - 以T2h的中值卖出").append('t2h_m')
       
        #获取T1,T2,T3 low/high预测值的排序结果
        self.___sort_predict_price()
        if self.predict_result not in self.Strategy:
            if len(self.predict_result)==7:
                self.predict_result = self.predict_result[:5]  # 去掉最后2位的low/high预测值
            if self.predict_result not in self.Strategy:
                print("ERROR: Strategy.___sort_predict_price()返回的预测结果不在Strategy中, 预测结果为: [%s]"%self.predict_result)
                exit()

    #*******************************************************************************************
    #   根据T1,T2,T3 low/high预测值, 生成买入价格
    #   返回值为买入价格, 如果不买则返回-1
    #*******************************************************************************************
    def get_buy_price(self):
        self.buy_price = self.get_specific_price(self.Strategy[self.predict_result][1])
        self.buy_price = get_mind_value(self.buy_price, self.predict.base_price)
        return self.buy_price 

    #*******************************************************************************************
    #   根据T1,T2,T3 low/high预测值, 生成卖出价格
    #   返回值为卖出价格, 如果不卖则返回-1
    #*******************************************************************************************
    def get_sell_price(self):
        self.sell_price = self.get_specific_price(self.Strategy[self.predict_result][3])
        self.sell_price = get_mind_value(self.sell_price, self.predict.base_price)
        return self.sell_price

    #*******************************************************************************************
    #   获取买入策略字符串
    #*******************************************************************************************
    def get_buy_strategy_str(self):
        self.buy_strategy_str = self.Strategy[self.predict_result][0]
        return self.buy_strategy_str

    #*******************************************************************************************
    #   获取卖出策略字符串
    #*******************************************************************************************    
    def get_sell_strategy_str(self):
        self.sell_strategy_str = self.Strategy[self.predict_result][2]
        return self.sell_strategy_str
    
    #*******************************************************************************************
    #   根据T1,T2,T3 low/high预测值, 生成买入/卖出价格
    #   返回值为买入/卖出价格, 如果不买/不卖则返回-1
    #*******************************************************************************************
    def get_specific_price(self, price_type):
        if price_type == 't1h_l':
            return self.predict.t1h_predict.plv
        elif price_type == 't1h_m':
            return self.predict.t1h_predict.average_value
        elif price_type == 't1h_h':
            return self.predict.t1h_predict.phv
        elif price_type == 't1l_l':
            return self.predict.t1l_predict.plv
        elif price_type == 't1l_m':
            return self.predict.t1l_predict.average_value
        elif price_type == 't1l_h':
            return self.predict.t1l_predict.phv
        elif price_type == 't2h_l':
            return self.predict.t2h_predict.plv
        elif price_type == 't2h_m':
            return self.predict.t2h_predict.average_value
        elif price_type == 't2h_h':
            return self.predict.t2h_predict.phv
        elif price_type == 't2l_l':
            return self.predict.t2l_predict.plv
        elif price_type == 't2l_m':
            return self.predict.t2l_predict.average_value
        elif price_type == 't2l_h':
            return self.predict.t2l_predict.phv
        elif price_type == 't3h_l':
            return self.predict.t3h_predict.plv
        elif price_type == 't3h_m':
            return self.predict.t3h_predict.average_value
        elif price_type == 't3h_h':
            return self.predict.t3h_predict.phv
        elif price_type == 't3l_l':
            return self.predict.t3l_predict.plv
        elif price_type == 't3l_m':
            return self.predict.t3l_predict.average_value
        elif price_type == 't3l_h':
            return self.predict.t3l_predict.phv
        return -1
