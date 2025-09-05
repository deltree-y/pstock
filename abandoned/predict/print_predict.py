# coding=utf-8
import os, sys
import numpy as np
from pathlib import Path

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.utils import get_predict_score, data_type_to_tv
from predict.predict_proc import PredictProc
from predict.cat import Cat

class PrintPredict():
    def __init__(self, model, X, Y, stock_datasets, model_file, data_type):
        self.model = model
        self.x = X
        self.y = Y
        self.stock_datasets = stock_datasets
        self.model_file = model_file
        if data_type not in ['t1l', 't1h', 't2l', 't2h', 't3l', 't3h']:
            print("ERROR: PrintPredict(): data_type must be one of ['t1l', 't1h', 't2l', 't2h', 't3l', 't3h']!")
            exit()
        self.data_type = data_type
        self.TV = data_type_to_tv(data_type)
        self.data_len = len(self.x)

    def PrintPredict(self):
        pass

    def PrintPredictEvaluate(self, is_only_print_score=True, is_write_file=True):
        Tn_x_score = 0
        for index in range(0,self.data_len):
            predict, accu = self.model.pred_data(self.x[index])
            pred_Tn_x = np.argmax(predict)
            real_Tn_x = np.argmax(self.y[index])
            pred_pp = PredictProc(cat_cls=Cat(classes=pred_Tn_x), data_type=self.data_type)
            real_pp = PredictProc(cat_cls=Cat(classes=real_Tn_x), data_type=self.data_type)
            if not is_only_print_score:
                print("INFO: real-[%s]"%real_pp.classes_to_description())
                print("INFO: pred-[%s] accu[%3.0f%%]"%(pred_pp.classes_to_description(),accu*100))
                print("***********************************************************************************")
            Tn_x_score += get_predict_score(pred_Tn_x, real_Tn_x)


        print("\n*****************************************")
        print("* INFO: model_file is <%s>"%self.model_file)
        print("* INFO: Tn xxx predict accuracy is [%.1f%%] *"%((Tn_x_score/self.data_len)*100))
        print("*****************************************")
        newest_dataset = self.stock_datasets.get_newest_dataset()
        pre, accu = self.model.pred_data(newest_dataset)
        pp = PredictProc(cat_cls=Cat(cat=pre),data_type=self.data_type, base_price=self.stock_datasets.tc.get_data_by_date(self.stock_datasets.newest_date).close)
        print("INFO: <%s> %s is <%s> accu[%3.0f%%]by[%s]\n"%(self.stock_datasets.newest_date, self.data_type, \
                                                             str(pp.classes_to_description_with_value()), accu*100, self.model_file))

        #fo = open("data\\predict.txt", "a+")
        #fo.write("[%s], predict time [%s] -\n"%(self.model_file, datetime.now()))
        #fo.write("predict accuracy: Tn_x[%.1f%%]"%((Tn_x_score/self.data_len)*100))
        #fo.write("\n\n")
        #fo.close()
