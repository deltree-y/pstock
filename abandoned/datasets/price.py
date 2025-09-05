#-*- coding:UTF-8 -*-
import sys
import os
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

#Price() - 用于记录/计算某日的价格数据，其中：
#   op - open price 开盘价格
#   cp - close price 闭市价格
#   lp - lowest price 最低价格
#   hp - highest price 最高价格
class Price():
    def __init__(self, op, cp, lp, hp):
        self.op = op
        self.cp = cp
        self.lp = lp
        self.hp = hp

    #判断预测的低值是否正确
    def is_lowest_hit(self, pre_llp=None, pre_hlp=None):
        if any([pre_llp is None, pre_hlp is None]):
            print("ERROR: Price().is_lowest_hit() must input all lower_low and higher_low price in predict.")
            sys.exit()
        if self.lp>=pre_llp and self.lp<=pre_hlp: #若比预测的低值高，比预测的高值低，则算是预测准了
            return True
        else:
            return False

    #判断预测的高值是否正确
    def is_highest_hit(self, pre_lhp=None, pre_hhp=None):
        if any([pre_lhp is None, pre_hhp is None]):
            print("ERROR: Price().is_highest_hit() must input all lower_high and higher_high price in predict.")
            sys.exit()
        if self.hp>=pre_lhp and self.hp<=pre_hhp: #若比预测的低值高，比预测的高值低，则算是预测准了
            return True
        else:
            return False

    #判断给定价格是否可以成功买入
    def is_good_buy(self, bp=None):
        if bp is None:
            print("ERROR: Price().is_good_buy() must input buy price.")
            sys.exit()
        #print("DEBUG: lp/hp - <%.2f/%.2f>"%(self.lp,self.hp))
        if bp >= self.lp:   #若买价比低值高，则算可成功买入
            return True
        else:
            return False

    #判断给定价格是否可以成功卖出
    def is_good_sell(self, sp=None):
        if sp is None:
            print("ERROR: Price().is_good_sell() must input buy price.")
            sys.exit()
        #print("DEBUG: lp/hp - <%.2f/%.2f>"%(self.lp,self.hp))
        if sp <= self.hp:   #若卖价比高值低，则算可成功卖出
            return True
        else:
            return False



if __name__ == "__main__":
    p = Price(45.28, 46.02, 45.1, 46.1)
    print(p.is_sell_success(46.1))
    print(p.is_lowest_hit(23,55))
