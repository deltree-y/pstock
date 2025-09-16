# codeing: utf-8
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 warmup_steps=0,
                 hold_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_begin(self, batch, logs=None):
        self.global_steps = self.model.optimizer.iterations.numpy()
        lr = self.get_lr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.learning_rates.append(lr)

    def get_lr(self):
        # 预热阶段
        if self.global_steps < self.warmup_steps:
            lr = self.learning_rate_base * (self.global_steps / self.warmup_steps)
        # 恒定学习率阶段
        elif self.global_steps < self.warmup_steps + self.hold_steps:
            lr = self.learning_rate_base
        # 余弦衰减阶段
        else:
            steps_since_hold = self.global_steps - self.warmup_steps - self.hold_steps
            cosine_steps = self.total_steps - self.warmup_steps - self.hold_steps
            progress = steps_since_hold / cosine_steps
            lr = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * progress))
        return lr


def auto_adjust_class_weights(y_pred, num_classes=6, min_weight=0.5, max_weight=5.0, power=1.0):
    """
    自动根据预测分布调整类别权重
    :param y_pred: 预测标签，一维数组，如[0,0,1,5,2...]
    :param num_classes: 类别总数
    :param min_weight: 最低权重限制
    :param max_weight: 最高权重限制
    :param power: 调整权重时的幂指数（可用于加剧或缓和权重差异，1.0为线性）
    :return: 类别权重列表
    """
    # 统计预测分布
    counts = np.bincount(y_pred, minlength=num_classes)
    total = counts.sum()
    props = counts / total

    # 计算权重: 预测比例越高，权重越低。权重 = 1/比例
    # 防止除0，可以加一个极小值
    weights = 1.0 / (props + 1e-6)
    # 可选：加大权重差异
    weights = weights ** power

    # 限制最大/最小权重
    weights = np.clip(weights, min_weight, max_weight)

    # 标准化权重，使平均为1（可选）
    weights = weights / np.mean(weights)

    # 输出权重
    print("自动调整的类别权重：", weights)
    return weights.tolist()

def confusion_based_weights(y_true, y_pred, num_classes=6, min_weight=0.5, max_weight=5.0):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    # 计算每个类别被误判为其它类别的总次数
    miscls_counts = cm.sum(axis=1) - np.diag(cm)
    # 被误判越多，权重越高
    weights = 1.0 + miscls_counts / (np.sum(miscls_counts)+1e-6) * (max_weight-1)
    weights = np.clip(weights, min_weight, max_weight)
    weights = weights / np.mean(weights)
    print(f"基于混淆矩阵调整的类别权重：{weights.tolist()}")
    return weights.tolist()