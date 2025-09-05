# coding=utf-8
import numpy as np
from keras.utils import to_categorical
from utils.const_def import NUM_CLASSES

#Cat用于处理分类, 其中：
#   pct_chg - 具体的数值如3.5/5/-1
#   cat     - 一个预测的数组, 其中最大值的位置为预测分类如：[0,0,1,0,0,0]/[1,3,6,3,8,0]
#   classes - 自行定义的预测类别分组, 如2/5/10
class Cat():
    def __init__(self, pct_chg=None, cat=None, classes=None, TV=None):
        if pct_chg is not None:
            self.pct_chg = pct_chg
            self.cat = self.___get_categorical(self.pct_chg, TV)
            self.classes = np.argmax(self.cat)
        elif cat is not None:
            self.cat = cat
            self.classes = np.argmax(self.cat)
        elif classes is not None:
            self.classes = classes
            self.cat = to_categorical(self.classes,num_classes=NUM_CLASSES)
        else:
            print("ERROR: Cat.init() - ALL inbound para is None!!!")
            exit()
        #print("DEBUG: cat is %s"%str(self.cat))
        #print("DEBUG: classes is %s"%str(self.classes))

    #返回股票变化率对应的0/1列表
    #如: 输入当num_classes=10时, 若当前变化率类别为2, 则返回[0,0,1,0,0,0,0,0,0,0]
    def ___get_categorical(self, pct_chg, TV):
        #print("DEBUG: self.__get_classes(pct_chg) - [%s]"%str(self.__get_classes(pct_chg)))
        return to_categorical(self.__get_classes(pct_chg, TV),num_classes=NUM_CLASSES)

    #get_classes()用于将具体的变化率转化为0-NUM_CLASSES之间的几个类别编号
    #返回值为具体的类别编号
    def __get_classes(self, pct_chg, TV):
        if pct_chg < float(TV[0]):
            classes = 0
        elif pct_chg <= float(TV[1]):
            classes = 1
        elif pct_chg <= float(TV[2]):
            classes = 2
        elif pct_chg <= float(TV[3]):
            classes = 3
        elif pct_chg <= float(TV[4]):
            classes = 4
        elif pct_chg <= float(TV[5]):
            classes = 5
        elif pct_chg <= float(TV[6]):
            classes = 6
        elif pct_chg <= float(TV[7]):
            classes = 7
        elif pct_chg <= float(TV[8]):
            classes = 8
        else:
            classes = 9
        return classes


    #cat - 一个预测的数组, 其中最大值的位置为预测分类如：[0,0,1,0,0,0]/[1,3,6,3,8,0]
    def get_cat(self):
        return self.cat
    
    #classes - 自行定义的预测类别分组, 如2/5/10
    def get_classes(self):
        return self.classes

