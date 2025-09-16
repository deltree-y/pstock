import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 y_true, y_pred 已经是一维的分箱标签数组，如 [0,1,2,2,5,...]
# 你可以把INFO count of vy_reg和y_pred_label的原始数据直接传进来

def plot_confusion(y_true, y_pred, num_classes=6):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    return cm