import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score

from model.utils import auto_adjust_class_weights, confusion_based_weights
from predicproc.predict import Predict
from utils.utils import print_ratio
from utils.utils import PredictType

# 假设 y_true, y_pred 已经是一维的分箱标签数组，如 [0,1,2,2,5,...]
# 你可以把INFO count of vy_reg和y_pred_label的原始数据直接传进来
def plot_confusion(y_true, y_pred, num_classes=6, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()
    return cm

def auto_adjust_class_weights(pred_raw, y_true, num_classes):
    y_pred = np.argmax(pred_raw, axis=1)
    auto_adjust_class_weights(y_pred, num_classes)
    confusion_based_weights(y_true, y_pred, num_classes)

def plot_confusion_by_model(pred_raw, y_true, num_classes=6, title="Confusion Matrix"):
    y_pred = np.argmax(pred_raw, axis=1)
    ret = plot_confusion(y_true, y_pred, num_classes=num_classes, title=title)
    return ret

def print_recall_score(pred_raw, y_true, predict_type):
    if predict_type.is_classify():
        y_pred = np.argmax(pred_raw, axis=1)
        print_ratio(y_pred, "y_pred_label")
        recall_score_list = recall_score(y_true, y_pred, average=None) 
        recall_score_list = [round(x, 3) for x in recall_score_list]    #保留三位小数
        macro_recall = round(recall_score(y_true, y_pred, average='macro'), 3)  #保留三位小数
        print(f"分类召回率: {recall_score_list}")
        print(f"宏召回率: {macro_recall}")
    elif predict_type.is_binary():
        y_pred = (pred_raw[:,0]>0.5).astype(int)
        print_ratio(y_pred, "y_pred_label")
        recall = round(recall_score(y_true, y_pred), 3)  #保留三位小数
        acc = round(accuracy_score(y_true, y_pred), 3)
        f1 = round(f1_score(y_true, y_pred), 3)
        print(f"二分类 准确率: {acc:.3f}, 召回率: {recall:.3f}, F1: {f1:.3f}")
    else:
        raise ValueError(f"print_recall_score() - Unknown predict_type: {predict_type}")

#基于给定的日期list,使用给定的数据集和模型进行预测并打印结果
def print_predict_result(t_list, ds, m, predict_type):
    for t0 in t_list:
        data, bp = ds.get_predictable_dataset_by_date(t0)
        #print("*************************************************************")
        #print(f"raw data is {data}")
        #print("*************************************************************\n")
        pred_scaled = m.model.predict(data, verbose=0)
        print(f"T0[{t0}]", end="")
        Predict(pred_scaled, bp, ds.bins1, ds.bins2, predict_type).print_predict_result()
    print("-------------------------------------------------------------")

