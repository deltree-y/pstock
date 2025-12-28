import os, sys, logging, warnings
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from keras import activations
from keras.models import Model
from keras.regularizers import l2
from keras.layers import (
    Input, LSTM, Bidirectional, Dropout, LayerNormalization,
    Dense, Add, Conv1D, GlobalAveragePooling1D, Multiply, Activation, Lambda
)
from model.base_model import BaseModel   # 新增
from model.history import LossHistory    # 仍可重用
from utils.utils import PredictType

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
    squeeze = Dense(ch // ratio, activation=activations.swish, name=f"{name_prefix}_fc1")(squeeze)
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
    shortcut = x
    # 原有一层 BiLSTM
    y = Bidirectional(
        LSTM(units, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        name=f"bilstm_{block_id}")(x)
    y = LayerNormalization(name=f"ln_{block_id}")(y)
    y = Dropout(dropout_rate, name=f"drop_{block_id}")(y)
    # 新增一层 BiLSTM
    #y = Bidirectional(
    #    LSTM(units, return_sequences=True),
    #    name=f"bilstm_{block_id}_2")(y)
    #y = LayerNormalization(name=f"ln_{block_id}_2")(y)
    #y = Dropout(dropout_rate, name=f"drop_{block_id}_2")(y)
    # SE 模块
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
    out = Activation(activations.swish, name=f"act_{block_id}")(out)
    return out

class ResidualLSTMModel(BaseModel):
    def __init__(self,
                 x=None, y=None,
                 test_x=None,test_y=None,
                 fn=None, p=2,
                 depth=3, base_units=32,use_se=True, se_ratio=8,
                 dropout_rate=0.2, l2_reg=1e-5,
                 loss_fn=None, class_weights=None, loss_type=None,
                 predict_type=PredictType.CLASSIFY
                 ):
        self.p = p
        self.depth = depth
        self.base_units = base_units
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.l2_reg = l2_reg
        self.loss_fn = loss_fn  # 保存传入的自定义损失（可为 None）

        if fn is not None:  #根据是否传入文件名来决定是否加载已有模型
            self.model = super().load(fn)
            self.history = LossHistory(predict_type=predict_type, test_x=test_x, test_y=test_y)
            #self.model = ResidualLSTMModel.load(fn, custom_objects=None)
            return

        super().__init__(
            x=x,
            y=y,
            test_x=test_x,
            test_y=test_y,
            loss_type=loss_type,
            class_weights=class_weights,
            predict_type=predict_type,
        )

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
        x_last = Dense(self.base_units * self.p * 2, activation=activations.swish,
                       kernel_regularizer=l2(self.l2_reg),
                       name="fc1")(x_last)
        #x_last = Dropout(self.dropout_rate, name="fc1_drop")(x_last)
        x_last = Dense(self.base_units  * 2, activation=activations.swish,
                       #kernel_regularizer=l2(self.l2_reg),
                       name="fc2")(x_last)
        # 输出头, 由基类实现, 根据 predict_type 自动选择
        outputs = self.build_output_head(x_last, self.predict_type)
        self.model = Model(inputs=inp, outputs=outputs)