import sys, os, logging
import numpy as np
from cat import RateCat
from utils.utils import PredictType
from utils.const_def import ACCU_RATE_THRESHOLD

class Predict():
    def __init__(self, predicted_data, base_price, predict_type: PredictType, bins1=None, bins2=None, threshold=0.5):
        self.predicted_data = predicted_data
        self.bp = base_price
        self.bins1, self.bins2 = bins1, bins2
        self.predict_type: PredictType = predict_type
        self.threshold = threshold
        self.is_classify = predict_type.is_classify()
        self.is_binary = predict_type.is_binary()
        self.is_regress = predict_type.is_regression()

        self.y1r = RateCat(one_hot=self.predicted_data[0,:], scale=self.bins1.bins, tag='T1L') if self.is_classify else None
        #self.y2r = RateCat(one_hot=self.predicted_data[1,:], scale=self.bins2.bins, tag='T2H')

        if self.is_classify:
            if bins1 is None:
                raise ValueError("分类任务需要传入 bins1.")
            if predicted_data.shape[-1] > 1:
                import numpy as np
                if np.allclose(predicted_data.sum(axis=-1), 1, atol=1e-4):
                    self.y1r = RateCat(one_hot=predicted_data[0,:], scale=self.bins1.bins, tag='T1L')
                else:
                    one_hot = np.zeros_like(predicted_data[0])
                    one_hot[np.argmax(predicted_data[0])] = 1
                    self.y1r = RateCat(one_hot=one_hot, scale=self.bins1.bins, tag='T1L')
            else:
                label = int(predicted_data[0,0])
                self.y1r = RateCat(label=label, scale=self.bins1.bins, tag='T1L')
        elif self.is_binary:
            self.prob = float(predicted_data[0,0])
            self.pred_label = int(self.prob > self.threshold)
        elif self.is_regress:
            self.pred_value = float(predicted_data[0,0])
        else:
            raise ValueError("未知的预测类型。")
    
    #打印预测结果
    def print_predict_result(self, desc="预测"):
        if self.is_classify:
            predict_list = [round(x, 3) for x in self.predicted_data[0]]
            confi = max(self.predicted_data[0]) * 100  # 多分类的置信率
            print(f"{desc}类别概率分布: {predict_list}")
            print(f"{desc}t0p[{self.bp}] t1l label[{self.y1r.get_label()}]({confi:.2f}%) 区间: [{self.y1r.get_rate_from_label('min')*100:.2f}%, {self.y1r.get_rate_from_label('max')*100:.2f}%], 均值: {self.y1r.get_rate_from_label('avg')*100:.2f}%, 预测价格: {self.bp * (1 + self.y1r.get_rate_from_label('avg')):.2f}")
        elif self.is_binary:
            if self.predict_type.is_t1_low() or self.predict_type.is_t2_low():
                symbol = ['<=', '> ']
            else:
                symbol = ['>=', '< ']
            label = f"{symbol[0]} {self.predict_type.val:.1f}%({self.bp*(1+self.predict_type.val/100):.2f})" if self.pred_label==1 else f"{symbol[1]} {self.predict_type.val:.1f}%({self.bp*(1+self.predict_type.val/100):.2f})"
            prob_rate = self.prob*100 if self.pred_label==1 else (1-self.prob)*100
            print(f"{desc}RAW<{self.prob:<.3f}>, T0价格[{self.bp:.2f}], {self.predict_type.label} {label}, 置信率:[{prob_rate:.2f}%]")
        elif self.is_regress:
            print(f"{desc}回归预测涨跌幅: {self.pred_value:.4f}, 预测价格: {self.bp * (1 + self.pred_value):.2f}")
        else:
            print("未知的预测类型，无法打印结果。")

    #打印预测结果VS真实值,返回字符串
    def get_predict_result_with_real_str(self, t0, bp, real_y):
        predict_wrong_str = ""
        low_high_symbol = "涨" if self.predict_type.is_high() else "跌"
        if self.is_classify:
            predict_list = [round(x, 3) for x in self.predicted_data[0]]
            confi = max(self.predicted_data[0]) * 100  # 多分类的置信率
            #print(f"预测类别概率分布: {predict_list}")
            try:
                #real_label = int(real_y[0,0]) if getattr(real_y, 'ndim', 0) and real_y.ndim > 1 else int(real_y[0])
                real_label = int(real_y)
            except Exception:
                real_label = None
            if real_label is not None:
                pred_label = int(self.y1r.get_label())
                pred_dot_str = f"{pred_label}" if pred_label == real_label else "x"
                if pred_label != real_label:
                    predict_wrong_str = (
                        f"T0[{t0}] - "
                        f"预测t0p[{self.bp:.2f}]" 
                        f"t1l label[{pred_label}]({confi:.2f}%), 区间: [{self.y1r.get_rate_from_label('min')*100:.2f}%, "
                        f"{self.y1r.get_rate_from_label('max')*100:.2f}%]({self.y1r.get_rate_from_label('avg')*100:.2f}%), "
                        f"预测价格: {self.bp * (1 + self.y1r.get_rate_from_label('avg')):.2f}"
                        )
        elif self.is_binary:
            if self.predict_type.is_t1_low() or self.predict_type.is_t2_low():
                symbol = ['<=', '> ']
            else:
                symbol = ['>=', '< ']
            label = f"{symbol[0]} {self.predict_type.val:.1f}%({self.bp*(1+self.predict_type.val/100):.2f})" if self.pred_label==1 else f"{symbol[1]} {self.predict_type.val:.1f}%({self.bp*(1+self.predict_type.val/100):.2f})"
            prob_rate = self.prob*100 if self.pred_label==1 else (1-self.prob)*100
            pred_result_str = f"" if self.pred_label==real_y else f" <--- [{self.pred_label}/{real_y}]预测错误!!!"
            pred_dot_str = f"{self.pred_label}" if self.pred_label==real_y else f"x"
            predict_wrong_str = (
                f"T0[{t0}] - "
                f"预测RAW<{self.prob:<.3f}>, "
                f"T0bp[{self.bp:.2f}], {self.predict_type.label} {label}, "
                f"置信率[{prob_rate:.2f}%] {pred_result_str}"
                ) if self.pred_label!=real_y else ""
        elif self.is_regress:
            pred_dot_str = "-"
            pred_dot_str = "o" if np.abs(self.pred_value - real_y*100) <=ACCU_RATE_THRESHOLD else "x"
            predict_wrong_str = (
                f"T0[{t0}] - "
                f"[真实/预测值]__{low_high_symbol}幅(差值) : "
                f"[{(real_y*bp+bp):<.2f}/{self.bp * (1 + self.pred_value/100):.2f}]  "
                f"{real_y*100: .2f}%/{self.pred_value: .2f}%({(real_y*100 - self.pred_value): .2f})"
                ) if pred_dot_str=="x" else ""
        else:
            pred_dot_str = "?"
            print("未知的预测类型，无法打印结果。")
        
        return pred_dot_str, predict_wrong_str


    @staticmethod
    #打印真实标签
    def from_real_label(real_y, base_price, predict_type:PredictType, bins1=None, bins2=None):
        import numpy as np
        real_y = np.asarray(real_y)
        if predict_type.is_classify():
            if bins1 is None:
                raise ValueError("分类任务需要传入 bins1.")
            label = int(real_y[0]) if real_y.ndim>0 else int(real_y)
            y1r = RateCat(label=label, scale=bins1.bins, tag='T1L')
            obj = Predict(np.zeros((1, bins1.n_bins+1)), base_price, predict_type, bins1, bins2)
            obj.y1r = y1r
            obj.is_classify = True
            return obj
        elif predict_type.is_binary():
            prob = float(real_y[0]) if real_y.ndim>0 else float(real_y)
            obj = Predict(np.array([[prob]]), base_price, predict_type)
            obj.is_binary = True
            obj.pred_label = int(prob)
            obj.prob = float(prob)
            return obj
        elif predict_type.is_regression():
            value = float(real_y[0]) if real_y.ndim>0 else float(real_y)
            obj = Predict(np.array([[value]]), base_price, predict_type)
            obj.is_regress = True
            obj.pred_value = value
            return obj
        else:
            raise ValueError("未知的预测类型。")


class RegPredict():
    def __init__(self, predicted_data, base_price, std_y=1, mean_y=0):
        self.predicted_data = predicted_data  # shape: [n, 1]
        self.bp = base_price
        self.std_y = std_y
        self.mean_y = mean_y

    def print_predict_result(self):
        pred_rate = self.predicted_data[0][0] * self.std_y + self.mean_y  # 直接取预测值
        pred_price = self.bp * (100 + pred_rate)/100
        logging.info(f"Predict base_price[{self.bp}] 预测涨跌幅[{pred_rate:.2f}%] 预测价格[{pred_price:.2f}]")