import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model.utils import auto_adjust_class_weights, confusion_based_weights
from predicproc.predict import Predict
from utils.utils import print_ratio

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

def plot_confusion_by_model(model, x, y_true, num_classes=6, title="Confusion Matrix"):
    y_pred_probs = model.model.predict(x)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print_ratio(y_pred, "y_pred_label")
    auto_adjust_class_weights(y_pred, num_classes)
    confusion_based_weights(y_true, y_pred, num_classes)
    ret = plot_confusion(y_true, y_pred, num_classes=num_classes, title=title)
    return ret

#基于给定的日期list,使用给定的数据集和模型进行预测并打印结果
def print_predict_result(t_list, ds, m):
    for t0 in t_list:
        print(f"Predict for T0[{t0}]")
        data, bp = ds.get_predictable_dataset_by_date(t0)
        pred_scaled = m.model.predict(data)
        print(f"Predict scaled result: {pred_scaled}")
        Predict(pred_scaled, bp, ds.bins1, ds.bins2).print_predict_result()
        print()

