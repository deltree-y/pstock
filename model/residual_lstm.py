import os, sys, logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout, Add, Concatenate, BatchNormalization, LayerNormalization, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from model.history import LossHistory
from keras.losses import Huber
from model.utils import WarmUpCosineDecayScheduler
from model.losses import mse_with_variance_push, direction_weighted_mse, custom_asymmetric_loss 
from keras.layers import (
    Input, LSTM, Bidirectional, Dropout, LayerNormalization,
    Dense, Add, Conv1D, GlobalAveragePooling1D, Multiply, Activation
)

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

def se_block(x, ratio=8, name_prefix="se"):
    """
    Squeeze-and-Excitation 简化实现:
    对时间维做 GAP -> 全连接降维 -> ReLU -> 全连接升维 -> Sigmoid -> 缩放
    """
    ch = x.shape[-1]
    if ch is None:
        return x
    squeeze = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    squeeze = Dense(ch // ratio, activation='relu', name=f"{name_prefix}_fc1")(squeeze)
    excite = Dense(ch, activation='sigmoid', name=f"{name_prefix}_fc2")(squeeze)
    excite = tf.expand_dims(excite, axis=1)
    return Multiply(name=f"{name_prefix}_scale")([x, excite])

def residual_bilstm_block(x,
                          units,
                          dropout_rate=0.2,
                          use_se=True,
                          se_ratio=8,
                          block_id=0,
                          l2_reg=1e-5):
    """
    单个残差块:
    输入: x (batch, time, features)
    主路径: BiLSTM(return_sequences=True) -> LayerNorm -> Dropout -> (SE 可选)
    shortcut投影: 若输入与输出通道不等，用1x1 Conv1D 做线性映射
    最终: Add + 激活
    """
    shortcut = x
    y = Bidirectional(
        LSTM(units,
             return_sequences=True,
             kernel_regularizer=l2(l2_reg)),
        name=f"bilstm_{block_id}"
    )(x)
    y = LayerNormalization(name=f"ln_{block_id}")(y)
    y = Dropout(dropout_rate, name=f"drop_{block_id}")(y)

    if use_se:
        y = se_block(y, ratio=se_ratio, name_prefix=f"se_{block_id}")

    in_ch = shortcut.shape[-1]
    out_ch = y.shape[-1]
    if (in_ch is not None) and (out_ch is not None) and (in_ch != out_ch):
        # 用1x1卷积投影 feature 维度
        shortcut = Conv1D(filters=out_ch,
                          kernel_size=1,
                          padding='same',
                          name=f"proj_{block_id}")(shortcut)

    out = Add(name=f"add_{block_id}")([y, shortcut])
    out = Activation('relu', name=f"act_{block_id}")(out)
    return out

class ResidualLSTMModel:
    """
    回归任务专用残差式 BiLSTM 模型
    """
    def __init__(self,
                 x=None,
                 y=None,
                 test_x=None,
                 test_y=None,
                 fn=None,
                 p=2,
                 depth=3,
                 base_units=32,
                 dropout_rate=0.2,
                 use_se=True,
                 se_ratio=8,
                 l2_reg=1e-5,
                 loss_fn=None
                 ):
        """
        p: 放大尺度(向后兼容)
        depth: 残差块个数
        base_units: 每个 BiLSTM 内部 units = base_units * p
        """
        if fn is not None:
            self.load(fn)
            self.model.summary()
            return

        if x is None or y is None:
            logging.error("ResidualLSTMModel init fail, need x & y or fn.")
            raise ValueError("No training data provided.")

        self.p = p
        self.x = x.astype('float32')
        self.y = y.astype('float32')
        self.test_x = test_x.astype('float32') if test_x is not None else None
        self.test_y = test_y.astype('float32') if test_y is not None else None

        self.history = LossHistory()
        self.depth = depth
        self.base_units = base_units
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.l2_reg = l2_reg
        self.loss_fn = loss_fn  # 保存传入的自定义损失（可为 None）

        logging.info(f"ResidualLSTMModel: input shape={self.x.shape}, y shape={self.y.shape}")
        self._build(self.x.shape[1:])

    def _build(self, input_shape):
        inp = Input(shape=input_shape, name="input")
        x = inp

        for i in range(self.depth):
            x = residual_bilstm_block(
                x,
                units=self.base_units * self.p,
                dropout_rate=self.dropout_rate,
                use_se=self.use_se,
                se_ratio=self.se_ratio,
                block_id=i,
                l2_reg=self.l2_reg
            )

        # 汇聚时间维（也可以尝试 attention / last timestep）
        # 这里使用最后时间步 + LN
        x_last = x[:, -1, :]
        x_last = LayerNormalization(name="ln_last")(x_last)
        x_last = Dropout(self.dropout_rate, name="drop_last")(x_last)

        # 回归头
        x_last = Dense(self.base_units * self.p, activation='relu',
                       kernel_regularizer=l2(self.l2_reg),
                       name="fc1")(x_last)
        x_last = Dropout(self.dropout_rate, name="fc1_drop")(x_last)
        x_last = Dense(self.base_units * self.p // 2, activation='relu',
                       kernel_regularizer=l2(self.l2_reg),
                       name="fc2")(x_last)
        out = Dense(1, activation='linear', name="output")(x_last)

        self.model = Model(inputs=inp, outputs=out, name="ResidualLSTMReg")
    

    def train(self, epochs=100, batch_size=32, learning_rate=0.001, patience=30):
        # Huber损失函数，对异常值更鲁棒
        loss_to_use = self.loss_fn if getattr(self, 'loss_fn', None) is not None else Huber(delta=0.1) #'mse'
        loss_to_use = custom_asymmetric_loss    # 使用自定义非对称损失函数
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss=loss_to_use, #Huber(delta=0.1),  # 使用Huber损失而不是MSE
            metrics=['mae']
        )        
        
        # 添加学习率调度和早停
        warmup_steps = int(0.1 * epochs)  # 10%的步数用于预热
        lr_scheduler = WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate,
            total_steps=epochs,
            warmup_steps=warmup_steps,
            hold_steps=int(0.05 * epochs)  # 5%的步数保持不变
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        
        start_time = datetime.now()
        self.history.set_para(epochs, start_time)
        
        # 添加所有callback
        callbacks = [self.history, lr_scheduler, early_stopping]
        
        self.model.fit(
            x=self.x,
            y=self.y,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y), 
            validation_freq=1, 
            callbacks=callbacks,
            epochs=epochs, 
            shuffle=True, 
            verbose=0  # 改为0以减少输出
        )
        
        spend_time = datetime.now() - start_time
        return "\n total spend:%.2f(h)/%.1f(m), %.1f(s)/epoc, %.2f(h)/10k"\
              %(spend_time.seconds/3600, spend_time.seconds/60, spend_time.seconds/epochs, 10000*(spend_time.seconds/3600)/epochs)

    def analyze(self, vx, vy_orig, mean_y, std_y, desc=""):
        """
        vx: 验证集特征
        vy_orig: 验证集真实标签(未标准化, 原百分比)
        mean_y, std_y: 训练集目标的均值/标准差(若训练使用了标准化)
        """
        pred_scaled = self.model.predict(vx, batch_size=2048).reshape(-1)
        y_pred = pred_scaled * std_y + mean_y
        # 统计
        real_mean, real_std = np.mean(vy_orig), np.std(vy_orig)
        pred_mean, pred_std = np.mean(y_pred), np.std(y_pred)
        collapse_ratio = pred_std / (real_std + 1e-9)
        direction_acc = np.mean(np.sign(y_pred) == np.sign(vy_orig))
        pearson = np.corrcoef(y_pred, vy_orig)[0, 1]
        try:
            from scipy.stats import spearmanr
            spearman = spearmanr(y_pred, vy_orig).correlation
        except Exception:
            spearman = np.nan
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        real_q = np.quantile(vy_orig, qs)
        pred_q = np.quantile(y_pred, qs)
        logging.info(f"[ANALYZE]{desc} real_mean={real_mean:.4f} real_std={real_std:.4f} "
                     f"pred_mean={pred_mean:.4f} pred_std={pred_std:.4f} collapse_ratio={collapse_ratio:.3f}")
        logging.info(f"[ANALYZE]{desc} dir_acc={direction_acc:.3f} pearson={pearson:.3f} spearman={spearman:.3f}")
        for qi, rq, pq in zip(qs, real_q, pred_q):
            logging.info(f"[ANALYZE]{desc} Q{qi*100:>4.0f} real={rq:.4f} pred={pq:.4f}")
        return {
            "real_mean": real_mean,
            "real_std": real_std,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "collapse_ratio": collapse_ratio,
            "direction_acc": direction_acc,
            "pearson": pearson,
            "spearman": spearman
        }

    def save(self, filename):
        try:
            self.model.save(filename)
            logging.info(f"\nmodel file saved -[{filename}]")
        except:
            logging.error("\nmodel file save failed!")
            exit()

    def load(self, filename):
        try:
            print("\nloading model file -[%s]..."%(filename),end="",flush=True)
            self.model = load_model(filename)
            print("complete!")
        except:
            logging.error("\nmodel file load failed!")
            exit()