# codeing: utf-8
import logging
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

    def on_epoch_begin(self, epoch, logs=None):
        #self.global_steps = self.model.optimizer.iterations.numpy()
        lr = self.get_lr(epoch)
        #if lr != tf.keras.backend.get_value(self.model.optimizer.lr):
        #    print(f" lr change from <{tf.keras.backend.get_value(self.model.optimizer.lr):.8f}> to <{lr:.8f}>",end='')
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.learning_rates.append(lr)

    def get_lr(self, epoch):
        # 预热阶段
        if epoch < self.warmup_steps:
            lr = self.learning_rate_base * (epoch / self.warmup_steps)
            self.model.learning_rate_status = "warmup"
            #print("\n(warmup)", end='')
        # 恒定学习率阶段
        elif epoch < self.warmup_steps + self.hold_steps:
            lr = self.learning_rate_base
            self.model.learning_rate_status = "hold"
            #print("\n(hold)", end='')
        # 余弦衰减阶段
        else:
            steps_since_hold = epoch - self.warmup_steps - self.hold_steps
            cosine_steps = self.total_steps - self.warmup_steps - self.hold_steps
            progress = steps_since_hold / cosine_steps
            lr = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * progress))
            self.model.learning_rate_status = "cosine"
            #print("\n(cosine decay)", end='')
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

def get_sample_weights(y, hard_mask, base_weight=1.0, hard_weight=3.0):
    """
    为hard样本增权，普通样本权重为base_weight，hard样本权重为hard_weight
    y: 标签 [N,]
    hard_mask: bool数组 [N,]，True代表hard样本
    返回: sample_weight数组 [N,]
    """
    sample_weight = np.full_like(y, base_weight, dtype=float)
    sample_weight[hard_mask] = hard_weight
    return sample_weight

def get_hard_samples(x, y, model, threshold=0.5):
    """
    找到置信度低于 threshold 的样本，为后续重点训练做准备
    x: 特征数据 [N, ...]
    y: 标签 [N,]
    model: 已训练好的 Keras 模型
    threshold: 置信度阈值，默认 0.5
    返回 (hard_x, hard_y) 置信度低的样本
    """
    # 得到每个样本预测的概率分布
    y_pred_prob = model.predict(x)  # shape: [N, num_classes]
    # 置信度=最大概率
    conf = np.max(y_pred_prob, axis=1)
    hard_mask = conf < threshold
    hard_x = x[hard_mask]
    hard_y = y[hard_mask]
    return hard_x, hard_y