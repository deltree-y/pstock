# coding: utf-8
import os, sys, logging
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, LayerNormalization, Activation, Conv1D, Lambda
from utils.utils import PredictType
from utils.const_def import NUM_CLASSES, IS_PRINT_MODEL_SUMMARY
from model.base_model import BaseModel                    # 新增
from model.history import LossHistory

class ResidualBlock(tf.keras.layers.Layer):
    """
    单层TCN残差块：两次扩张卷积->归一化->激活->Dropout->Add残差
    支持因果卷积与通道数自动投影
    """
    def __init__(self, nb_filters, kernel_size, dilation_rate, block_id, dropout_rate=0.1, l2_reg=1e-5, causal=True, **kwargs):
        #K.clear_session()
        #super(ResidualBlock, self).__init__(name=f"residual_block_{block_id}", **kwargs)
        super().__init__(**kwargs)
        self.causal = causal
        self.block_id = block_id
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding='causal' if causal else 'same',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                            name=f"c{block_id}")
        self.norm1 = LayerNormalization(name=f"ln1_{block_id}")
        self.activation1 = Activation('relu', name=f"act1_{block_id}")
        self.dropout1 = Dropout(dropout_rate, name=f"drop1_{block_id}")

        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding='causal' if causal else 'same',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                            name=f"conv2_{block_id}")
        self.norm2 = LayerNormalization(name=f"ln2_{block_id}")
        self.activation2 = Activation('relu', name=f"act2_{block_id}")
        self.dropout2 = Dropout(dropout_rate, name=f"drop2_{block_id}")

        self.use_projection = False
        self.projection = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "nb_filters": self.conv1.filters,
            "kernel_size": self.conv1.kernel_size[0],
            "dilation_rate": self.conv1.dilation_rate[0],
            "block_id": self.block_id,     # 新增
            "dropout_rate": self.dropout1.rate,
            "l2_reg": self.conv1.kernel_regularizer.l2 if self.conv1.kernel_regularizer else 0.0,
            "causal": self.causal,
        })
        return config
    
    def build(self, input_shape):
        # 若输入通道和输出通道不同，用1x1卷积投影
        if input_shape[-1] != self.conv1.filters:
            self.use_projection = True
            self.projection = Conv1D(filters=self.conv1.filters, kernel_size=1, padding='same', name=f"proj_{self.block_id}")
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation1(x)
        if training:
            x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        if training:
            x = self.dropout2(x)
        if self.use_projection:
            residual = self.projection(residual)
        out = Add(name=f"add_{self.block_id}")([x, residual])
        return out, x  # 返回主路径和skip分支

class ResidualTCNModel(BaseModel):    # 继承 BaseModel
    """
    主流残差TCN模型，支持因果卷积/多stack堆叠/skip分支加和
    """
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, p=2,
                 nb_filters=64, kernel_size=8, nb_stacks=1, dilations=None, dropout_rate=0.1,
                 l2_reg=1e-5, causal=True,
                 class_weights=None, loss_type=None,
                 predict_type=PredictType.CLASSIFY
                 ):
        self.p = p
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations or [1, 2, 4, 8, 16, 32]
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.causal = causal
        if fn is not None:
            # 载入需要自定义层
            self.model = type(self).load(fn, custom_objects={"ResidualBlock": ResidualBlock})
            self.history = LossHistory(predict_type=predict_type, test_x=test_x, test_y=test_y)
            return
        super().__init__(x=x, y=y, test_x=test_x, test_y=test_y,
                         loss_type=loss_type, class_weights=class_weights,
                         predict_type=predict_type)
        logging.info(f"ResidualTCN input shape: {self.x.shape}, output shape: {self.y.shape}")

    def _build(self, input_shape):
        inputs = Input(shape=input_shape)
        x = inputs
        skip_connections = []
        for stack in range(self.nb_stacks):
            for dilation in self.dilations:
                block_id = f"{stack}_{dilation}"
                block = ResidualBlock(self.nb_filters, self.kernel_size, dilation, block_id,
                                      self.dropout_rate, self.l2_reg, causal=self.causal)
                x, skip = block(x)
                skip_connections.append(skip)
        x = Add()(skip_connections) if len(skip_connections) > 1 else skip_connections[0]
        x = x[:, -1, :]
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Activation('relu')(x)
        outputs = self.build_output_head(x, self.predict_type)   # 共用输出头
        self.model = Model(inputs=inputs, outputs=outputs)
