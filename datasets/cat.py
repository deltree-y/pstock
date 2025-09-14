# coding=utf-8
import os, sys, logging
import numpy as np
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.const_def import NUM_CLASSES, T1L_SCALE, T2H_SCALE
from utils.utils import setup_logging

#父类
#cond1 - in<label, num_classes>, out<one_hot>
#cond2 - in<one_hot>, out<label, num_classes>
class Cat():
    def __init__(self, label=None, one_hot=None, num_classes=NUM_CLASSES):
        self.label, self.one_hot, self.num_classes = label, one_hot, num_classes

        if label is not None and num_classes is not None:
            self.label = label
            self.one_hot = self.get_one_hot_fa()
        elif one_hot is not None:
            self.one_hot = one_hot
            self.num_classes = self.one_hot.shape[0]
            self.label = self.get_label_fa()
        else:
            logging.error("Cat.__init__() - label, one_hot and num_classes should not all be None!")
            exit()

    def get_one_hot_fa(self):
        if self.one_hot is not None:
            return self.one_hot
        elif self.label is not None and self.num_classes is not None:
            self.one_hot = np.eye(self.num_classes)[self.label]
        else:
            logging.error("Cat.get_one_hot() - label and one_hot should not both be None!")
            exit()
        return self.one_hot

    def get_label_fa(self):
        if self.label is not None:
            return self.label
        elif self.one_hot is not None:
            self.label = np.argmax(self.one_hot, axis=0)
        else:
            logging.error("Cat.get_label() - label and one_hot should not both be None!")
            exit()
        return self.label
    

#子类
#cond1 - in<rate, scale>, out<label, one_hot, num_classes>
#cond2 - in<label, scale>, out<rate, one_hot, num_classes>
#cond3 - in<one_hot, scale>, out<label, rate, num_classes>
class RateCat(Cat):
    def __init__(self, rate=None, label=None, one_hot=None, scale=None, tag=None, method='avg', right=False):
        self.scale , self.tag, self.right = scale, tag, right
        if self.scale is None:
            logging.error("RateCat.__init__() - scale should not be None!")
            exit()

        if rate is not None:
            self.rate = rate
            self.num_classes = len(scale) + 1
            self.label = self.get_label_from_rate(right)
            super().__init__(label=self.label, num_classes=self.num_classes)
        elif label is not None:
            self.label = label
            self.num_classes = len(scale) + 1
            self.rate = self.get_rate_from_label(method)
            super().__init__(label=self.label, num_classes=self.num_classes)
        elif one_hot is not None:
            self.one_hot = one_hot
            super().__init__(one_hot=self.one_hot)
            self.rate = self.get_rate_from_label(method)
    
    #根据变化率及刻度表, 输出编号
    #示例 - 
    #输入<-0.323>输出<1>, 输入<1.823>输出<9>, 
    def get_label_from_rate(self, right=False):
        if self.rate is not None and self.scale is not None:
            return np.digitize(self.rate, bins=self.scale)#, right=right)
        else:
            logging.error("RateCat.get_label_from_rate() - rate and scale should not be None!")
            exit()
    
    #根据编号.刻度表.提取方法, 输出变化率
    #示例 - 
    #输入<1,'min'>输出<-0.928>, 输入<1,'max'>输出<-0.913>
    def get_rate_from_label(self, method='avg'):
        #logging.info(f"RateCat.get_rate_from_label() - label={self.label}, method={method}, scale={self.scale}")
        if self.label is not None and self.scale is not None:
            if method == 'avg':
                if self.label == 0:
                    sim = self.scale[0] + (self.scale[0]-self.scale[1])
                    return np.mean([self.scale[0], sim])
                elif self.label == self.num_classes - 1:
                    sim = self.scale[-1] + (self.scale[-1] - self.scale[-2])
                    return np.mean([self.scale[-1], sim])
                else:
                    return np.mean(self.scale[self.label-1:self.label+1])
            elif method == 'max':
                if self.label == self.num_classes-1:
                    return self.scale[-1] + (self.scale[-1] - self.scale[-2])
                else:
                    return self.scale[self.label]
            elif method == 'min':
                if self.label == 0:
                    return self.scale[0] + (self.scale[0]-self.scale[1])
                else:
                    return self.scale[self.label-1]
            else:
                logging.error("RateCat.get_rate_from_label() - unknown method!")
                exit()
        else:
            logging.error("RateCat.get_rate_from_label() - label and scale should not both be None!")
            exit()

    def get_one_hot(self):
        return self.one_hot
    
    def get_label(self):
        return self.label
    
    def get_rate(self):
        return self.rate

if __name__ == "__main__":
    setup_logging()
    #print(RateCat(rate=1.115,scale=T1L_SCALE).get_label())
    #for i in range(20):
    #    print("%.3f,%.3f,%.3f"%(100*RateCat(label=i,method='min',scale=T1L_SCALE).get_rate(),\
    #                            100*RateCat(label=i,method='avg',scale=T1L_SCALE).get_rate(),\
    #                            100*RateCat(label=i,method='max',scale=T1L_SCALE).get_rate()))
    print(RateCat(rate=-0.01,scale=T1L_SCALE).get_one_hot())
