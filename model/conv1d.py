import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Dropout,
    Add, GlobalAveragePooling1D, Dense, Multiply, LayerNormalization, Lambda
)
from keras.regularizers import l2
from utils.utils import PredictType
from model.base_model import BaseModel
from model.history import LossHistory

def se_block(x, ratio=8, name_prefix="se"):
    ch = x.shape[-1]
    if ch is None:
        return x
    squeeze = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    squeeze = Dense(ch // ratio, activation="relu", name=f"{name_prefix}_fc1")(squeeze)
    excite = Dense(ch, activation='sigmoid', name=f"{name_prefix}_fc2")(squeeze)
    excite = tf.expand_dims(excite, axis=1)
    return Multiply(name=f"{name_prefix}_scale")([x, excite])

def residual_conv_block(x, filters, kernel_size=5, dropout_rate=0.2, l2_reg=1e-5, use_se=True, se_ratio=8, block_id=0):
    shortcut = x
    y = Conv1D(filters=filters, kernel_size=kernel_size, padding="same",
               kernel_regularizer=l2(l2_reg), name=f"conv1_{block_id}")(x)
    y = BatchNormalization(name=f"bn1_{block_id}")(y)
    y = Activation('relu', name=f"relu1_{block_id}")(y)
    y = Dropout(dropout_rate, name=f"drop1_{block_id}")(y)
    y = Conv1D(filters=filters, kernel_size=kernel_size, padding="same",
               kernel_regularizer=l2(l2_reg), name=f"conv2_{block_id}")(y)
    y = BatchNormalization(name=f"bn2_{block_id}")(y)
    y = Activation('relu', name=f"relu2_{block_id}")(y)
    y = Dropout(dropout_rate, name=f"drop2_{block_id}")(y)
    if use_se:
        y = se_block(y, ratio=se_ratio, name_prefix=f"se_{block_id}")
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding="same", name=f"proj_{block_id}")(shortcut)
    y = Add(name=f"add_{block_id}")([y, shortcut])
    y = Activation('relu', name=f"out_relu_{block_id}")(y)
    return y

class Conv1DResModel(BaseModel):
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, 
                 filters=64, kernel_size=5, depth=4, 
                 dropout_rate=0.2, l2_reg=1e-5, use_se=True, se_ratio=8,
                 class_weights=None, loss_type=None, 
                 predict_type=PredictType.CLASSIFY,
                 ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_se = use_se
        self.se_ratio = se_ratio

        if fn is not None:
            # 直接加载已保存模型（无自定义层，无需 custom_objects）
            self.model = type(self).load(fn)
            self.history = LossHistory(predict_type=predict_type, test_x=test_x, test_y=test_y)
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
        inp = Input(shape=input_shape)
        x = inp

        # 多层残差 Conv1D
        for i in range(self.depth):
            x = residual_conv_block(
                x,
                filters=self.filters,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
                use_se=self.use_se,
                se_ratio=self.se_ratio,
                block_id=i,
            )

        x = GlobalAveragePooling1D(name="gap")(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg), name="fc1")(x)
        x = Dropout(self.dropout_rate, name="fc1_drop")(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg), name="fc2")(x)
        x = Dropout(self.dropout_rate, name="fc2_drop")(x)

        # 共用基类输出头
        outputs = self.build_output_head(x, self.predict_type)
        self.model = Model(inputs=inp, outputs=outputs)