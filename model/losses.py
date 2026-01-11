import tensorflow as tf
from utils.utils import PredictType

def robust_mse_with_clip(clip=5.0, alpha=1.0):
    """
    对大误差做裁剪/降权的MSE：
      - 先对误差做截断 clip（单位：百分点）
      - 误差越大权重越低：1 / (1 + (err/clip)**alpha)
    """
    def loss_fn(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        err = tf.abs(y_true_f - y_pred_f)
        clipped = tf.minimum(err, clip)
        weight = 1.0 / (1.0 + tf.pow(err / clip, alpha))
        return tf.reduce_mean(weight * tf.square(clipped))
    loss_fn.__name__ = f"robust_mse_clip{clip}_a{alpha}"
    return loss_fn

def mse_with_variance_push(var_floor_ratio: float = 0.5, penalty_weight: float = 0.05, eps: float = 1e-9):
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

def custom_asymmetric_loss(y_true, y_pred):
    from keras import backend as K
    error = y_pred - y_true
    # 上涨预测为下跌的惩罚权重更大
    weights = K.cast(K.greater(y_true, 0), K.floatx()) * 1.5 + 1.0
    return K.mean(K.square(error) * weights, axis=-1)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # y_true shape: (batch_size, 1) 或 (batch_size,)
        y_true = tf.cast(y_true, tf.int32)
        if len(y_true.shape) == 2:  # 如果是 (batch_size, 1)
            y_true = tf.squeeze(y_true, axis=-1)  # 变成 (batch_size,)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # shape: (batch_size, num_classes)
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_factor = alpha * tf.pow(1. - p_t, gamma)
        return tf.reduce_mean(focal_factor * cross_entropy)
    return loss

def binary_focal_loss(gamma=2.0, alpha=0.25, from_logits=False):
    """
    标准 Binary Focal Loss（支持按类 alpha_t）:
      alpha_t = alpha      if y=1
              = 1 - alpha  if y=0

    pt = p        if y=1
       = 1 - p    if y=0

    loss = - alpha_t * (1 - pt)^gamma * log(pt)

    说明：
    - 你当前版本把 alpha 统一乘在所有样本上，会导致 alpha 调参意义失真。
    - 这个实现会让 alpha 真正变成“正类权重”，更可控。
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # shape 兼容 (batch, 1) / (batch,)
        if len(y_true.shape) == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 2 and y_pred.shape[-1] == 1:
            y_pred = tf.squeeze(y_pred, axis=-1)

        # logits/prob 兼容
        if from_logits:
            # 直接用 sigmoid 得到 prob
            p = tf.nn.sigmoid(y_pred)
        else:
            p = y_pred

        eps = tf.keras.backend.epsilon()
        p = tf.clip_by_value(p, eps, 1.0 - eps)

        # pt: 预测为真实类别的概率
        pt = tf.where(tf.equal(y_true, 1.0), p, 1.0 - p)

        # alpha_t: 按类加权
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)

        # focal factor
        focal_factor = tf.pow(1.0 - pt, gamma)

        # focal loss
        loss_val = -alpha_t * focal_factor * tf.math.log(pt)
        return tf.reduce_mean(loss_val)

    return loss

def confidence_penalty_loss(base_loss_fn, lam=0.2):
    def loss(y_true, y_pred):
        # 基本 BCE
        base = base_loss_fn(y_true, y_pred)
        # conf (y_pred 为概率)
        conf = tf.abs(y_pred - 0.5) * 2  # 置信度 [0,1]
        # 是否预测正确
        y_pred_label = tf.cast(y_pred > 0.5, tf.int32)
        y_true_cast = tf.cast(y_true, tf.int32)
        correct = tf.cast(tf.equal(y_pred_label, y_true_cast), tf.float32)
        # Encourage correct/high confidence, penalize wrong/high confidence
        penalty = lam * tf.reduce_mean((1 - correct) * conf)
        reward  = lam * tf.reduce_mean(correct * (1 - conf))
        return base + penalty + 0.5*reward
    return loss


# 根据输入值及预测类型来选择损失函数
def get_loss(loss_type, predict_type:PredictType):
    if predict_type.is_regression():
        if loss_type == 'robust_mse':
            return robust_mse_with_clip()
        # 回归默认使用MSE
        return 'mse' if loss_type is None else loss_type

    if loss_type == 'focal_loss':
        if predict_type.is_classify():
            loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        elif predict_type.is_binary():
            loss_fn = binary_focal_loss(gamma=5.0, alpha=0.25)
            #raise ValueError("Unsupported predict_type for focal_loss.")
    elif loss_type == 'confidence_penalty_loss':
        if predict_type.is_binary():
            base_loss = tf.keras.losses.BinaryCrossentropy()
            loss_fn = confidence_penalty_loss(base_loss, lam=0.2)
        else:
            raise ValueError("confidence_penalty_loss only supports binary classification.")
    else:
        if predict_type.is_classify():
            loss_fn = 'sparse_categorical_crossentropy'
        elif predict_type.is_binary():
            loss_fn = 'binary_crossentropy'#tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)#'binary_crossentropy'
        else:
            raise ValueError("Unsupported predict_type for classification model.")
    return loss_fn
