# codeing: utf-8
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.utils import ModelType
from utils.const_def import BASE_DIR, MODEL_DIR

# 学习率调度器
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

    # 每个epoch开始时调用
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.get_lr(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.learning_rates.append(lr)

    # 获取当前学习率
    def get_lr(self, epoch):
        # 预热阶段
        if epoch < self.warmup_steps:
            lr = self.learning_rate_base * (epoch / self.warmup_steps)
            self.model.learning_rate_status = "wup"

        # 恒定学习率阶段
        elif epoch < self.warmup_steps + self.hold_steps:
            lr = self.learning_rate_base
            self.model.learning_rate_status = "hld"
            
        # 余弦衰减阶段
        else:
            steps_since_hold = epoch - self.warmup_steps - self.hold_steps
            cosine_steps = self.total_steps - self.warmup_steps - self.hold_steps
            progress = steps_since_hold / cosine_steps
            lr = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * progress))
            self.model.learning_rate_status = "cos"
        return lr

# 自动调整类别权重，预测比例越高，权重越低
def auto_adjust_class_weights(y_pred, num_classes=6, min_weight=0.5, max_weight=5.0, power=1.0):
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

# 基于混淆矩阵调整类别权重
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

# 获取样本权重，给 hard 样本更高的权重
def get_sample_weights(y, hard_mask, base_weight=1.0, hard_weight=3.0):
    sample_weight = np.full_like(y, base_weight, dtype=float)
    sample_weight[hard_mask] = hard_weight
    return sample_weight

# 获取模型预测的 hard 样本
def get_hard_samples(x, y, y_pred_raw, predict_type, threshold=0.5):
    if predict_type.is_classify():
        # 置信度=最大概率
        conf = np.max(y_pred_raw, axis=1)
        hard_mask = conf < threshold
    elif predict_type.is_binary():
        # 置信度=离0.5的距离
        if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 1:
            y_pred_raw = y_pred_raw.reshape(-1)
        conf = np.abs(y_pred_raw - 0.5) * 2  # 归一到 [0,1]
        hard_mask = conf < threshold
    hard_x = x[hard_mask]
    hard_y = y[hard_mask]
    return hard_x, hard_y

def get_model_file_name(stock_code, model_type, predict_type, feature_type,suffix=""):
    if suffix != "":
        model_fn = os.path.join(BASE_DIR, MODEL_DIR, f"{stock_code}_{model_type}_{predict_type}_{feature_type.short_name}_{suffix}.h5")
    else:
        model_fn = os.path.join(BASE_DIR, MODEL_DIR, f"{stock_code}_{model_type}_{predict_type}_{feature_type.short_name}.h5")
    return model_fn


def load_model_by_params(stock_code, model_type, predict_type, feature_type, suffix=""):
    model_fn = get_model_file_name(stock_code, model_type, predict_type, feature_type, suffix=suffix)

    from model.residual_lstm import ResidualLSTMModel
    from model.residual_tcn import ResidualTCNModel
    from model.transformer import TransformerModel
    from model.conv1d import Conv1DResModel
    
    if model_type == ModelType.RESIDUAL_LSTM:
        model = ResidualLSTMModel(fn=model_fn, predict_type=predict_type)
    elif model_type == ModelType.RESIDUAL_TCN:
        model = ResidualTCNModel(fn=model_fn, predict_type=predict_type)
    elif model_type == ModelType.TRANSFORMER:
        model = TransformerModel(fn=model_fn, predict_type=predict_type)
    elif model_type == ModelType.CONV1D:
        model = Conv1DResModel(fn=model_fn, predict_type=predict_type)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model

