# codeing: utf-8
import tensorflow as tf
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


def direction_weighted_mse(y_true, y_pred):
    # 基础MSE
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # 方向一致性权重
    same_direction = tf.cast(tf.math.sign(y_true) == tf.math.sign(y_pred), tf.float32)
    direction_weight = 0.7 + 0.3 * same_direction  # 方向正确时给予更低权重
    
    return direction_weight * mse
