# coding: utf-8
"""
Custom losses used by training scripts.

Provides mse_with_variance_push(var_floor_ratio=0.5, penalty_weight=0.1)
which returns a callable loss(y_true, y_pred) compatible with tf.keras.
"""
import tensorflow as tf

def mse_with_variance_push(var_floor_ratio: float = 0.5, penalty_weight: float = 0.05, eps: float = 1e-9):
    """
    Return a loss function that is MSE plus a penalty if prediction std is less than var_floor_ratio * std(y_true).

    Parameters:
      var_floor_ratio: fraction of true-batch-std to require (e.g. 0.5 means require pred_std >= 0.5 * true_std)
      penalty_weight: weight of the penalty term added to MSE
      eps: small constant to avoid div-by-zero

    Behavior:
      loss = mse + penalty_weight * relu(var_floor - pred_std)
      where var_floor = var_floor_ratio * stop_gradient(std(y_true))
    """
    def loss_fn(y_true, y_pred):
        # ensure float32 tensors
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # base mse (mean over batch & output dims)
        mse = tf.reduce_mean(tf.square(y_true_f - y_pred_f))

        # predicted std (over batch). use sample std (tf.math.reduce_std)
        pred_std = tf.math.reduce_std(y_pred_f)  # scalar
        true_std = tf.math.reduce_std(y_true_f)  # scalar

        # var floor derived from true batch std, use stop_gradient so it's not trainable
        var_floor = var_floor_ratio * tf.stop_gradient(true_std)

        # penalty: if pred_std < var_floor, penalize the gap
        penalty = tf.nn.relu(var_floor - pred_std)

        # combine
        return mse + penalty_weight * penalty

    # set name so it appears in model.summary / logs
    loss_fn.__name__ = f"mse_varpush_r{var_floor_ratio}_w{penalty_weight}"
    return loss_fn

def direction_weighted_mse(y_true, y_pred):
    # 基础MSE
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # 方向一致性权重
    same_direction = tf.cast(tf.math.sign(y_true) == tf.math.sign(y_pred), tf.float32)
    direction_weight = 0.7 + 0.3 * same_direction  # 方向正确时给予更低权重
    
    return direction_weight * mse
