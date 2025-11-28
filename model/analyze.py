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

def plot_l2_loss_curves(history_dict, epochs, save_path=None):
    plt.figure(figsize=(10, 6))
    for l2_reg, record in history_dict.items():
        plt.plot(record['val_loss'], label=f'l2={l2_reg}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Curve for Different l2_reg')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def auto_adjust_class_weights(pred_raw, y_true, num_classes):
    y_pred = np.argmax(pred_raw, axis=1)
    auto_adjust_class_weights(y_pred, num_classes)
    confusion_based_weights(y_true, y_pred, num_classes)

def plot_confusion_by_model(pred_raw, y_true, num_classes=6, title="Confusion Matrix"):
    y_pred = np.argmax(pred_raw, axis=1)
    ret = plot_confusion(y_true, y_pred, num_classes=num_classes, title=title)
    return ret

def print_recall_score(pred_raw, y_true, predict_type, threshold=0.5):
    if predict_type.is_classify():
        y_pred = np.argmax(pred_raw, axis=1)
        print_ratio(y_pred, "预测数据")
        recall_score_list = recall_score(y_true, y_pred, average=None) 
        recall_score_list = [round(x, 3) for x in recall_score_list]    #保留三位小数
        macro_recall = round(recall_score(y_true, y_pred, average='macro'), 3)  #保留三位小数
        print(f"分类召回率: {recall_score_list}")
        print(f"宏召回率: {macro_recall}")
        return macro_recall
    elif predict_type.is_binary():
        y_pred = (pred_raw[:,0]>threshold).astype(int)
        print(f"-"*60)
        print_ratio(y_true, "真实数据")
        print_ratio(y_pred, "预测数据")
        recalls = recall_score(y_true, y_pred, average=None)
        f1s = f1_score(y_true, y_pred, average=None)
        acc = round(accuracy_score(y_true, y_pred), 3)
        print(f"-"*60)
        print(f"二分类 准确率: {acc:.3f}")
        print(f"类别0 召回率: {recalls[0]:.3f}, 类别1 召回率: {recalls[1]:.3f}, 平均召回率: {(recalls[0]+recalls[1])/2:.3f}")
        print(f"类别0 F1分数: {f1s[0]:.3f}, 类别1 F1分数: {f1s[1]:.3f}, 平均F1分数: {(f1s[0]+f1s[1])/2:.3f}")
        print(f"-"*60)
        return (recalls[0]+recalls[1])/2

    else:
        raise ValueError(f"print_recall_score() - Unknown predict_type: {predict_type}")

