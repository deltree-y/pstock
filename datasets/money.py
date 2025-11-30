#-*- coding:UTF-8 -*-
import sys
import os
from pathlib import Path
from datetime import datetime
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))


#Funds() - 用于记录/计算资金，其中：
#   ia - init amount 初始资金
#   ca - current amount 当前资金
class Funds():
    def __init__(self, init_amount=0):
        self.ia = init_amount       #初始资金
        self.ca = self.ia           #当前资金
        self.capital = init_amount  #本金
        self.quantity = 0           #当前所持股票数量
        self.half_quantity = 0
        self.buy_date = None        #买入日期
        self.cur_buy_cost = 0      #当前所持股票的成本价

        self.buy_fee = 0.0001954   #买入手续费0.02%
        self.sell_fee = 0.0006954  #卖出手续费0.07%

    #**************************************************************************************
    #deposit() - 增加资金
    def deposit(self, amount):
        self.ca += amount
        self.capital += amount

    #**************************************************************************************
    #withdraw() - 取出资金
    #   返回值 - 
    #               -1 取出失败，金额不变
    #                0 取出成功，当前资金/本金均会变化
    def withdraw(self, amount):
        if amount > self.ca:
            print("ERROR: Found.withdraw() - withdraw is bigger than current amount.")
            return -1
        self.ca -= amount
        self.capital -= amount
        return 0
    
    #**************************************************************************************
    #buy_stock() - 购买股票，其中-
    #   unit_price  - 单股价格
    #   quantity    - 购买数量(若购买一手，则此处为100)
    #   返回值 - 
    #               -1 购买失败，资金不变
    #               <%d> 购买成功，返回购买数量，当前资金/所持股票数量均会产生对应变化
    def buy_stock(self, unit_price, quantity, date=None):
        #print("\nDEBUG: Funds().buy_stock() before_buy, self.quantity[%d], self.ca[%d], try quantity[%d]"%(self.quantity,self.ca,quantity))
        if not self.is_enough_to_buy(unit_price, quantity):   #如果购买所需花费大于当前资金，则购买失败
            print("INFO: [%s]Buy at[%.2f] failed. COST[%s] is BIGGER than current amount[%s]."%(date, unit_price, \
                                                                                                   format(int(unit_price*quantity+unit_price*quantity*self.buy_fee),","), \
                                                                                                    format(int(self.ca),",")))
            return -1
        self.buy_date = date if date is not None else datetime.now().strftime("%Y%m%d")
        self.quantity += quantity
        self.ca -= unit_price*quantity
        self.ca -= unit_price*quantity*self.buy_fee #扣手续费
        self.cur_buy_cost = unit_price*quantity + unit_price*quantity*self.buy_fee
        return quantity

    #**************************************************************************************
    #buy_max() - 在当前价格下将现金最大限度全部购买为股票
    #   返回值 - 
    #               <%d> 购买股票数量
    def buy_max(self, unit_price, date=None, is_print=True):
        if not self.is_enough_to_buy(unit_price, 100):  #如果连一手都不够买的话，直接返回0
            print("INFO: [%s]buy_max  FAILED. NO MONEY."%date)
            return 0
        if self.quantity != 0:
            print("INFO: [%s]buy_max  FAILED. stock in-hand."%date)
            return 0
        buy_hand = self.ca // ((1+self.buy_fee)*unit_price*100)
        ret = self.buy_stock(unit_price, 100*buy_hand, date)
        if ret != -1:
            if is_print:
                print("INFO: [%s]buy_max  at[%.2f]! stock/amount is <%7s>/[￥ %s]"%(date, unit_price, \
                                                                                    format(int(self.get_stock_quantity()),","),\
                                                                                    format(int(self.get_total_amount(unit_price)),",")))
        return ret
    
    #**************************************************************************************
    #get_buy_max_quantity() - 计算在当前价格下最大可购买股票数量
    def get_buy_max_quantity(self, unit_price):
        if not self.is_enough_to_buy(unit_price, 100):  #如果连一手都不够买的话，直接返回0
            return 0
        if self.quantity != 0:
            return 0
        buy_hand = self.ca // ((1+self.buy_fee)*unit_price*100)
        return int(buy_hand*100)

    #**************************************************************************************
    #buy_half() - 在当前价格下将现金的一半购买为股票
    #   返回值 - 
    #               <%d> 购买股票数量
    def buy_half(self, unit_price, date=None):
        if not self.is_enough_to_buy(unit_price, 100):  #如果连一手都不够买的话，直接返回0
            return 0
        buy_hand = (self.ca/2) // (unit_price*100) if self.quantity == 0 else self.half_quantity
        self.half_quantity = buy_hand*100 if self.quantity == 0 else self.half_quantity
        ret = self.buy_stock(unit_price, 100*buy_hand, date)
        if ret != -1:
            print("INFO: [%s]buy_half at[%.2f]! stock/amount is <%7s>/[￥ %s]"%(date, unit_price, \
                                                                                format(int(self.get_stock_quantity()),","),\
                                                                                format(int(self.get_total_amount(unit_price)),",")))
        return ret

    #**************************************************************************************
    #sell_stock() - 卖出股票，其中-
    #   unit_price  - 单股价格
    #   quantity    - 购买数量(若购买一手，则此处为100)
    #   返回值 - 
    #               -1 卖出失败，资金不变
    #               <%d> 卖出成功，返回卖出数量，当前资金/所持股票数量均会产生对应变化
    def sell_stock(self, unit_price, quantity, date=None):
        if quantity > self.quantity:
            print("INFO: [%s]Sell stock failed. sell quantity is BIGGER than hold quantity."%date)
            return -1
        if self.buy_date == date and self.buy_date is not None:
            print("INFO: [%s]Sell stock failed. sell and buy cannot in the same day."%date)
            return -1
        self.quantity -= quantity
        self.ca += unit_price*quantity
        self.ca -= unit_price*quantity*self.sell_fee    #扣手续费
        return quantity
    
    #**************************************************************************************
    #sell_all() - 在当前价格下卖掉所有股票
    #   返回值 -    <%d>卖出股票数量
    def sell_all(self, unit_price, date=None, is_print=True):
        if self.get_stock_quantity() == 0:  #如果没有股票则不执行任何
            print("INFO: [%s]Sell stock failed. currnt quantity is ZERO."%date)
            return 0
        sell_quantity = self.get_stock_quantity()
        sell_ret = self.sell_stock(unit_price, sell_quantity, date)
        if sell_ret > 0:
            profit = ((sell_ret*unit_price - sell_ret*unit_price*self.sell_fee) - self.cur_buy_cost)/self.cur_buy_cost
            if is_print:
                print("INFO: [%s]sell_all at[%.2f]! stock/amount is <%7s>/[￥ %s]. profit[%+-.1f%%]"%(date, unit_price, \
                                                                                                    format(int(self.get_stock_quantity()),","),\
                                                                                                    format(int(self.get_total_amount(unit_price)),","),\
                                                                                                    profit*100))
        else:
            print("ERROR: sell_all FAILED.")
        return sell_quantity

    #**************************************************************************************
    #sell_half() - 在当前价格下卖掉占一半资金的股票
    #   返回值 -    <%d>卖出股票数量
    def sell_half(self, unit_price, date=None):
        if self.get_stock_quantity() == 0:  #如果没有股票则不执行任何
            return 0
        sell_quantity = self.half_quantity
        if self.sell_stock(unit_price, sell_quantity, date) > 0:
            print("INFO: [%s]sell_half at[%.2f]! stock/amount is <%7s>/[￥ %s]"%(date, unit_price, \
                                                                                 format(int(self.get_stock_quantity()),","),\
                                                                                 format(int(self.get_total_amount(unit_price)),",")))
        return sell_quantity

    #**************************************************************************************
    #get_profit() - 取当前利润值
    #   返回值 - 
    #               <%d> 当前股价下的利润额
    def get_profit(self, unit_price):
        return self.get_total_amount(unit_price) - self.capital
    
    #**************************************************************************************
    #get_profit_margin() - 取当前利润率
    #   返回值 - 
    #               <%d> 100*当前股价下的利润额/本金
    #               -1   若本金为0则返回-1
    def get_profit_margin(self, unit_price):
        if self.capital == 0:
            print("WARNNING: Found().get_profit_margin() - current capital is ZERO!")
            return -1
        return 100*self.get_profit(unit_price)/self.capital
    
    #**************************************************************************************
    #get_stock_amount() - 取当前股票市值
    #   返回值 - 
    #               <%d> 当前股价下的股票市值
    def get_stock_amount(self, unit_price):
        return self.quantity * unit_price

    #**************************************************************************************
    #get_stock_quantity() - 取当前股票数量
    #   返回值 - 
    #               <%d> 当前所持股票数量
    def get_stock_quantity(self):
        return self.quantity

    #**************************************************************************************
    #get_total_amount() - 取当前股票与现金总价值
    #   返回值 - 
    #               <%d> 当前股价下的股票市值+当前现金
    def get_total_amount(self, unit_price):
        return self.get_stock_amount(unit_price) + self.ca

    #**************************************************************************************    
    #is_enough_to_buy() - 计算是否有足够资金购买给定数量股票
    #   返回值 - 
    #               True 够买
    #               True 不够买
    def is_enough_to_buy(self, unit_price, quantity):
        if (unit_price*quantity + unit_price*quantity*self.buy_fee) > self.ca:   #如果购买所需花费大于当前资金
            return False
        else:
            return True



if __name__ == "__main__":
    f = Funds(100000)
    f.buy_stock(50,1000)
    #print("DEBUG: profit margin is <%.2f%%>"%f.get_profit_margin(35))
    print("DEBUG: stock quantity is <%s>, current amount is <%d>"%(format(int(f.get_stock_quantity()),","), f.ca))
    f.sell_stock(50, 1000)
    #print("DEBUG: profit margin is <%.2f%%>"%f.get_profit_margin(45))
    print("DEBUG: stock quantity is <%d>, current amount is <%d>"%(f.get_stock_quantity(), f.ca))
    f.buy_max(45)
    print("DEBUG: stock quantity is <%d>, current amount is <%d>"%(f.get_stock_quantity(), f.ca))
    f.sell_all(50)
    print("DEBUG: stock quantity is <%d>, current amount is <%d>"%(f.get_stock_quantity(), f.ca))
    print("DEBUG: profit margin is <%.2f%%>"%f.get_profit_margin(50))
    f.sell_all(50)
    print("DEBUG: stock quantity is <%d>, current amount is <%d>"%(f.get_stock_quantity(), f.ca))
    print("DEBUG: profit margin is <%.2f%%>"%f.get_profit_margin(50))
