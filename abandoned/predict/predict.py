# coding=utf-8
import time
from model.model import StockModel
from predict.predict_proc import PredictProc
from predict.cat import Cat
from utils.const_def import BASE

#PredictList用于记录多预测模型, 并进行处理, 其中 - 
#   ts_code - 预测的股票代码
#   model_name_list - 预测模型的名称列表
#   bp - 用于预测的最近收盘数据
#   T  - 用于预测的最近日期(T0)
class Predict():
    def __init__(self, ts_code, model_name_list, input_data=None, base_price=None, t_date=None):
        self.ts_code = ts_code
        self.model_name_list = model_name_list
        self.t1l_model, self.t1h_model, self.t2l_model, self.t2h_model, self.t3l_model, self.t3h_model = self.get_models()
        if t_date is not None:
            self.t0 = t_date
        if base_price is not None:
            self.base_price = base_price
        if input_data is not None:
            self.get_predicts(input_data)

    #********************************************************************************************************************
    #   set_T()用于设置预测日期 
    #  注意：预测日期为字符串格式, 如'2023-10-01'
    #  返回值为None
    # ********************************************************************************************************************    
    def set_T(self, date):
        self.t0 = date

    #********************************************************************************************************************
    #   set_base_price()用于设置预测的基准价格
    #   注意：基准价格为float类型, 如100.0
    #   返回值为None
    # ********************************************************************************************************************
    def set_base_price(self, base_price):
        self.base_price = base_price

    #********************************************************************************************************************
    #   get_models()用于获取预测模型
    #   返回值为6个StockModel对象, 分别对应t1l, t1h, t2l, t2h, t3l, t3h
    #   注意：模型文件名为ts_code + '_' + model_name + '.h5'
    #   如：ts_code为'000001', model_name_list为['t1l', 't1h', 't2l', 't2h', 't3l', 't3h']
    #   则模型文件名为'000001_t1l.h5', '000001_t1h.h5', '000001_t2l.h5', '000001_t2h.h5', '000001_t3l.h5', '000001_t3h.h5'
    #   注意：模型文件存放在BASE目录下
    def get_models(self):
        return StockModel(filename=BASE + self.ts_code + '_' + self.model_name_list[0] + ".h5"),\
                StockModel(filename=BASE + self.ts_code + '_' + self.model_name_list[1] + ".h5"),\
                StockModel(filename=BASE + self.ts_code + '_' + self.model_name_list[2] + ".h5"),\
                StockModel(filename=BASE + self.ts_code + '_' + self.model_name_list[3] + ".h5"),\
                StockModel(filename=BASE + self.ts_code + '_' + self.model_name_list[4] + ".h5"),\
                StockModel(filename=BASE + self.ts_code + '_' + self.model_name_list[5] + ".h5")
    
    #********************************************************************************************************************
    #   get_predicts()用于获取预测结果
    #   返回值为6个预测结果, 分别对应t1l, t1h, t2l, t2h, t3l, t3h
    #   注意：预测结果为模型的原始预测结果, 即[0.1, 0.2, 0.3, 0.4]等
    #   注意：预测结果为numpy数组, 需要转换为Cat类才能使用
    #********************************************************************************************************************
    def get_predicts(self, input_data):
        self.input_data = input_data
        sleep_time = 0.01  # 每次预测之间的睡眠时间, 防止模型加载过快导致数据不一致
        t1l_pre, t1l_accu = self.t1l_model.pred_data(self.input_data)
        time.sleep(sleep_time)
        t1h_pre, t1h_accu = self.t1h_model.pred_data(self.input_data)
        time.sleep(sleep_time)
        t2l_pre, t2l_accu = self.t2l_model.pred_data(self.input_data)
        time.sleep(sleep_time)
        t2h_pre, t2h_accu = self.t2h_model.pred_data(self.input_data)
        time.sleep(sleep_time)
        t3l_pre, t3l_accu = self.t3l_model.pred_data(self.input_data)
        time.sleep(sleep_time)
        t3h_pre, t3h_accu = self.t3h_model.pred_data(self.input_data)
        time.sleep(sleep_time)
        #print("DEBUG: predict raw-\n%s\n%s\n%s\n%s\n%s\n%s\n"%(t1l_pre, t1h_pre, t2l_pre, t2h_pre, t3l_pre, t3h_pre))

        self.t1l_predict_raw, self.t1h_predict_raw, self.t2l_predict_raw, self.t2h_predict_raw, self.t3l_predict_raw, self.t3h_predict_raw = \
                                                            t1l_pre, t1h_pre, t2l_pre, t2h_pre, t3l_pre, t3h_pre
        self.t1l_accu, self.t1h_accu, self.t2l_accu, self.t2h_accu, self.t3l_accu, self.t3h_accu = \
                                                            t1l_accu, t1h_accu, t2l_accu, t2h_accu, t3l_accu, t3h_accu
        self.t1l_predict = PredictProc(cat_cls=Cat(cat=self.t1l_predict_raw), base_price=self.base_price, data_type='t1l')
        self.t1h_predict = PredictProc(cat_cls=Cat(cat=self.t1h_predict_raw), base_price=self.base_price, data_type='t1h')
        self.t2l_predict = PredictProc(cat_cls=Cat(cat=self.t2l_predict_raw), base_price=self.base_price, data_type='t2l')
        self.t2h_predict = PredictProc(cat_cls=Cat(cat=self.t2h_predict_raw), base_price=self.base_price, data_type='t2h')
        self.t3l_predict = PredictProc(cat_cls=Cat(cat=self.t3l_predict_raw), base_price=self.base_price, data_type='t3l')
        self.t3h_predict = PredictProc(cat_cls=Cat(cat=self.t3h_predict_raw), base_price=self.base_price, data_type='t3h')

        return t1l_pre, t1h_pre, t2l_pre, t2h_pre, t3l_pre, t3h_pre

    #********************************************************************************************************************
    #   get_predict_string()用于获取预测结果字符串  
    #   返回值为字符串, 格式为INFO: <ts_code> T1 low is <预测结果> by[模型名称]
    #   注意：预测结果为PredictProc类的classes_to_description_with_value()方法
    #*********************************************************************************************************************
    def get_predict_string(self):
        ret = ""
        ret = ret + "INFO: <%s> T1 low  <%s> accu[%3.0f%%] by[%s]\n"%(self.t0, str(self.t1l_predict.classes_to_description_with_value()), \
                                                                   self.t1l_accu*100, self.model_name_list[0])
        ret = ret + "INFO: <%s> T1 high <%s> accu[%3.0f%%] by[%s]\n"%(self.t0, str(self.t1h_predict.classes_to_description_with_value()), \
                                                                   self.t1h_accu*100, self.model_name_list[1])
        ret = ret + "INFO: <%s> T2 low  <%s> accu[%3.0f%%] by[%s]\n"%(self.t0, str(self.t2l_predict.classes_to_description_with_value()), \
                                                                   self.t2l_accu*100, self.model_name_list[2])
        ret = ret + "INFO: <%s> T2 high <%s> accu[%3.0f%%] by[%s]\n"%(self.t0, str(self.t2h_predict.classes_to_description_with_value()), \
                                                                   self.t2h_accu*100, self.model_name_list[3])
        ret = ret + "INFO: <%s> T3 low  <%s> accu[%3.0f%%] by[%s]\n"%(self.t0, str(self.t3l_predict.classes_to_description_with_value()), \
                                                                   self.t3l_accu*100, self.model_name_list[4])
        ret = ret + "INFO: <%s> T3 high <%s> accu[%3.0f%%] by[%s]"  %(self.t0, str(self.t3h_predict.classes_to_description_with_value()), \
                                                                   self.t3h_accu*100, self.model_name_list[5])
        return ret

    #********************************************************************************************************************
    #   print_predict_string()用于打印预测结果
    #   注意：打印结果为INFO: <ts_code> T1 low is <预测结果> by[模型名称]
    #********************************************************************************************************************
    def print_predict_string(self):
        print("%s"%self.get_predict_string())

    #********************************************************************************************************************
    #   get_t1_low_predict()用于获取T1 low预测结果  
    #   返回值为一个元组, 包含T1 low预测的低值和高值
    #   注意：返回值为PredictProc类的plv和phv属性   
    #*********************************************************************************************************************
    def get_t1_low_predict(self):
        return self.t1l_predict.plv, self.t1l_predict.phv
    
    #********************************************************************************************************************
    #   get_t1_high_predict()用于获取T1 high预测结果
    #   注意：预测结果为PredictProc类的plv和phv属性
    #   返回值为一个元组, 包含T1 high预测的低值和高值
    #   注意：返回值为PredictProc类的plv和phv属性
    #*********************************************************************************************************************
    def get_t1_high_predict(self):
        return self.t1h_predict.plv, self.t1h_predict.phv

    #********************************************************************************************************************
    #   get_t2_low_predict()用于获取T2 low预测结果
    #   返回值为一个元组, 包含T2 low预测的低值和高值
    #   注意：返回值为PredictProc类的plv和phv属性   
    #*********************************************************************************************************************
    def get_t2_low_predict(self):
        return self.t2l_predict.plv, self.t2l_predict.phv

    #********************************************************************************************************************
    #   get_t2_high_predict()用于获取T2 high预测结果    
    #   返回值为一个元组, 包含T2 high预测的低值和高值
    #   注意：返回值为PredictProc类的plv和phv属性
    #*********************************************************************************************************************
    def get_t2_high_predict(self):
        return self.t2h_predict.plv, self.t2h_predict.phv

    #********************************************************************************************************************
    #   get_t3_low_predict()用于获取T3 low预测结果
    #   返回值为一个元组, 包含T3 low预测的低值和高值
    #   注意：返回值为PredictProc类的plv和phv属性
    #*********************************************************************************************************************
    def get_t3_low_predict(self):
        return self.t3l_predict.plv, self.t3l_predict.phv

    #********************************************************************************************************************
    #   get_t3_high_predict()用于获取T3 high预测结果
    #   返回值为一个元组, 包含T3 high预测的低值和高值
    #   注意：返回值为PredictProc类的plv和phv属性
    #*********************************************************************************************************************
    def get_t3_high_predict(self):
        return self.t3h_predict.plv, self.t3h_predict.phv

