# coding=utf-8
import sys,os

from pathlib import Path
from utils.utils import data_type_to_tv, get_symbol
from utils.const_def import NUM_CLASSES

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

#PredictProc用于根据输入的数据, 处理后输出符合格式的文本信息, 其中：
#   cat_cls = Cat()类型, 用于输入未处理的预测数值
#   base_price = 目标股票的预测前价格
#   data_type = 预测数据类型, 如't1l', 't1h', 't2l', 't2h', 't3l', 't3h'
class PredictProc():
    def __init__(self, cat_cls=None, base_price=None, data_type=None):

        if data_type is not None:
            self.data_type = data_type
            self.TV = data_type_to_tv(data_type)
            if 'l' in self.data_type:
                self.symbol = "-"
            elif 'h' in self.data_type:
                self.symbol = "+"
            else:
                print("ERROR: Predict.init() - data_type is not valid, please check your input parameters!")
        else:
            print("ERROR: Predict.init() - data_type is None, please check your input parameters!")

        if cat_cls is not None:
            self.cat = cat_cls

        if base_price is not None:
            self.bp = base_price
            self.plv, self.phv = self.classes_to_value()   #self.plv/phv(Predict Low/High Value )为具体的预测数值
            self.average_value = (self.plv + self.phv) / 2  #self.average_value为预测的平均值
            self.pltv, self.phtv = self.classes_to_TV()  #self.pltv/phtv(Predict Low/High TV)为预测涨跌幅百分比
            self.average_tv = (self.pltv + self.phtv) / 2  #self.average_tv为预测涨跌幅的平均值
        else:
            pass
            #print("WARNING: PredictProc() - [base_price] is empty!")


    #classes_to_description()用于将类别编号转换为类别描述的字符串
    #返回值为具体的字符串
    def classes_to_description(self):
        TV = self.TV
        if self.cat.classes == 0 :
            return ("%2d,  (-∞     , %.2f)"%(self.cat.get_classes(), TV[0]))
        if self.cat.classes == NUM_CLASSES - 1 :
            return ("%2d,  (%.2f ,    +∞)"%(self.cat.get_classes(), TV[NUM_CLASSES-2]))
        for i in range(1, NUM_CLASSES-1):
            if self.cat.classes == i:
                return ("%2d,  [%.2f , %.2f]"%(self.cat.get_classes(), TV[i-1],TV[i]))


    #classes_to_description()用于将类别编号转换为类别描述的字符串
    #返回值为具体的字符串
    def classes_to_description_with_value(self):
        TV = self.TV
        if self.cat.classes == 0 :
            symbols = get_symbol(TV[0]*2, TV[0])
            return ("%s,  (-∞           , %.2f(%+.1f%%))"%(symbols,self.bp+self.bp*0.01*TV[0],TV[0]))
        if self.cat.classes == NUM_CLASSES - 1 :
            symbols = get_symbol(TV[NUM_CLASSES-2], TV[NUM_CLASSES-2]*2)
            return ("%s,  (%.2f(%+.1f%%) ,           +∞)"%(symbols,self.bp+self.bp*0.01*TV[NUM_CLASSES-2],TV[NUM_CLASSES-2]))
        for i in range(1, NUM_CLASSES-1):
            if self.cat.classes == i:
                symbols = get_symbol(TV[i-1], TV[i])
                return ("%s,  [%.2f(%+.1f%%) , %.2f(%+.1f%%)]"%(symbols,self.bp+self.bp*0.01*TV[i-1],TV[i-1],self.bp+self.bp*0.01*TV[i],TV[i]))


    #classes_to_TV()用于将类别编号转换为具体的预测值范围
    #返回值为一个元组, 包含低值和高值(涨跌幅百分比)
    #如：输入0, 返回(-10, TV[0])
    def classes_to_TV(self):
        TV = self.TV
        if self.cat.classes == 0 :
            if 'h' in self.data_type:
                return TV[0]*2, TV[0] #特殊处理：如果是高值预测, 则高低值返回TV[0]*2和TV[0]
            else:
                return -10, TV[0]   #正常返回
        elif self.cat.classes == NUM_CLASSES - 1 :
            if 'l' in self.data_type:
                return TV[NUM_CLASSES-2], TV[NUM_CLASSES-2]*2 #特殊处理：如果是低值预测, 则高低值返回TV[NUM_CLASSES-2]和TV[NUM_CLASSES-2]*2
            else:
                return TV[NUM_CLASSES-2], 10    #正常返回

        for i in range(1, NUM_CLASSES-1):
            if self.cat.classes == i:
                return TV[i-1], TV[i]
            
        print("ERROR: PredictProc()::classes_to_TV() - classes!!![%s]!!! not in range."%(str(self.cat.classes)))
        exit()


    #classes_to_value()用于将类别编号转换为具体的预测值
    #返回值为具体的预测数字, 一般为高低值的均值
    def classes_to_value(self):
        low, high = self.classes_to_TV()
        return self.bp+self.bp*0.01*(low), self.bp+self.bp*0.01*(high)
    
    
    #返回一个数字列表中最大数字对应的序号
    #如：输入[0,2,3,4,1], 返回3
    def categorical_to_description(self):
        TV = self.TV
        return self.classes_to_description(self.cat.get_cat(), TV)





