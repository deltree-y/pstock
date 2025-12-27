import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Add, GlobalAveragePooling2D, Dense, Multiply, Lambda, Flatten
from keras.regularizers import l2
from utils.utils import PredictType
from model.base_model import BaseModel
from model.history import LossHistory

def se_block_2d(x, ratio=8, name_prefix="se"):
    ch = x.shape[-1]
    if ch is None:
        return x
    squeeze = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    squeeze = Dense(max(ch // ratio, 1), activation="relu", name=f"{name_prefix}_fc1")(squeeze)
    excite = Dense(ch, activation='sigmoid', name=f"{name_prefix}_fc2")(squeeze)
    excite = Lambda(lambda z: tf.expand_dims(tf.expand_dims(z, 1), 1))(excite)
    return Multiply(name=f"{name_prefix}_scale")([x, excite])

def residual_conv2d_block(x, filters, kernel_size=(3,3), dropout_rate=0.2, l2_reg=1e-5, use_se=True, se_ratio=8, block_id=0):
    shortcut = x
    y = Conv2D(filters=filters, kernel_size=kernel_size, padding="same",
               kernel_regularizer=l2(l2_reg), name=f"conv1_{block_id}")(x)
    y = BatchNormalization(name=f"bn1_{block_id}")(y)
    y = Activation('relu', name=f"relu1_{block_id}")(y)
    y = Dropout(dropout_rate, name=f"drop1_{block_id}")(y)
    y = Conv2D(filters=filters, kernel_size=kernel_size, padding="same",
               kernel_regularizer=l2(l2_reg), name=f"conv2_{block_id}")(y)
    y = BatchNormalization(name=f"bn2_{block_id}")(y)
    y = Activation('relu', name=f"relu2_{block_id}")(y)
    y = Dropout(dropout_rate, name=f"drop2_{block_id}")(y)
    if use_se:
        y = se_block_2d(y, ratio=se_ratio, name_prefix=f"se_{block_id}")
    # 投影shortcut通道维度对齐
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters=filters, kernel_size=1, padding="same", name=f"proj_{block_id}")(shortcut)
    y = Add(name=f"add_{block_id}")([y, shortcut])
    y = Activation('relu', name=f"out_relu_{block_id}")(y)
    return y

class Conv2DResModel(BaseModel):
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, 
                 filters=32, kernel_size=(3,3), depth=3,
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
        x = inp  # [batch, window, feature, channel]
        for i in range(self.depth):
            x = residual_conv2d_block(
                x,
                filters=self.filters,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
                use_se=self.use_se,
                se_ratio=self.se_ratio,
                block_id=i,
            )
        # 聚合空间维：平均或flatten
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg), name="fc1")(x)
        x = Dropout(self.dropout_rate, name="fc1_drop")(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg), name="fc2")(x)
        x = Dropout(self.dropout_rate, name="fc2_drop")(x)
        outputs = self.build_output_head(x, self.predict_type)
        self.model = Model(inputs=inp, outputs=outputs)