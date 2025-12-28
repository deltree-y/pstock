# coding=utf-8
import os, sys, logging
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, MultiHeadAttention, Add, Flatten, Lambda, GlobalAveragePooling1D, Concatenate
from keras.regularizers import l2
from model.base_model import BaseModel                    # 新增
from utils.utils import PredictType
from model.history import LossHistory

# --- 优化: Time2Vec 层 (学习型时间编码) ---
class Time2Vec(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_regularizer=None, **kwargs):  # 增加**kwargs
        super(Time2Vec, self).__init__(**kwargs)                        # 传递**kwargs
        self.output_dim = output_dim
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        # input_shape: (batch, seq_len, 1)
        self.w_linear = self.add_weight(name='w_linear',
                                        shape=(1, 1),
                                        initializer='uniform',
                                        trainable=True)
        self.b_linear = self.add_weight(name='b_linear',
                                        shape=(1, 1),
                                        initializer='uniform',
                                        trainable=True)
        
        self.w_periodic = self.add_weight(name='w_periodic',
                                          shape=(1, self.output_dim - 1),
                                          initializer='uniform',
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)
        self.b_periodic = self.add_weight(name='b_periodic',
                                          shape=(1, self.output_dim - 1),
                                          initializer='uniform',
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, seq_len, 1)
        # Linear term: w*t + b
        v_linear = self.w_linear * inputs + self.b_linear 
        # Periodic term: sin(w*t + b)
        v_periodic = tf.math.sin(tf.matmul(inputs, self.w_periodic) + self.b_periodic)
        
        return tf.concat([v_linear, v_periodic], axis=-1)

    def get_config(self):  # MODIFIED: add get_config
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):  # MODIFIED: for compatibility
        config["kernel_regularizer"] = tf.keras.regularizers.deserialize(config["kernel_regularizer"])
        return cls(**config)


# --- 优化: Deep & Cross Network (DCN) 层 ---
class CrossNet(tf.keras.layers.Layer):
    def __init__(self, layer_num=2, reg=1e-5, **kwargs):               # 增加**kwargs
        super(CrossNet, self).__init__(**kwargs)                       # 传递**kwargs
        self.layer_num = layer_num
        self.reg = reg

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W = [self.add_weight(name=f'w_{i}', shape=(dim,), initializer='glorot_uniform',
                                  regularizer=l2(self.reg), trainable=True) for i in range(self.layer_num)]
        self.b = [self.add_weight(name=f'b_{i}', shape=(dim,), initializer='zeros',
                                  regularizer=l2(self.reg), trainable=True) for i in range(self.layer_num)]
        super(CrossNet, self).build(input_shape)

    def call(self, inputs):
        x0 = inputs
        xl = inputs
        for i in range(self.layer_num):
            # x_{l+1} = x0 * (x_l . w_l) + b_l + x_l
            # Dot product along last dim
            xw = tf.reduce_sum(xl * self.W[i], axis=-1, keepdims=True)
            xl = x0 * xw + self.b[i] + xl
        return xl

    def get_config(self):  # MODIFIED: add get_config
        config = super().get_config()
        config.update({
            "layer_num": self.layer_num,
            "reg": self.reg,
        })
        return config

    @classmethod
    def from_config(cls, config):  # MODIFIED: for compatibility
        return cls(**config)


def positional_encoding(length, depth):
    """
    优化建议1: 添加位置编码来增强序列位置信息
    """
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (seq, depth)
    
    # 将正弦应用于偶数索引，余弦应用于奇数索引
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate, l2_reg=1e-5, block_id=0, use_gating=False):
    # Self-attention with stronger regularization
    attn_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=d_model // num_heads,  # 按头数平均分配维度
        kernel_regularizer=l2(l2_reg),
        name=f"mha_{block_id}"
    )(x, x)
    attn_output = Dropout(dropout_rate, name=f"attn_dropout_{block_id}")(attn_output)
    
    # 第一个残差连接
    x = Add(name=f"add1_{block_id}")([x, attn_output])
    x = LayerNormalization(epsilon=1e-6, name=f"ln1_{block_id}")(x)
    
    # 前馈网络
    ff_input = x
    ff_output = Dense(ff_dim, activation='gelu', kernel_regularizer=l2(l2_reg), name=f"ff1_{block_id}")(x)
    ff_output = Dropout(dropout_rate, name=f"ff_dropout1_{block_id}")(ff_output)
    
    # 优化建议2: 可选的门控机制
    if use_gating:
        gate = Dense(ff_dim, activation='sigmoid', kernel_regularizer=l2(l2_reg), name=f"gate_{block_id}")(x)
        ff_output = tf.multiply(ff_output, gate)
    
    ff_output = Dense(d_model, kernel_regularizer=l2(l2_reg), name=f"ff2_{block_id}")(ff_output)
    ff_output = Dropout(dropout_rate, name=f"ff_dropout2_{block_id}")(ff_output)
    
    # 第二个残差连接
    x = Add(name=f"add2_{block_id}")([ff_input, ff_output])
    x = LayerNormalization(epsilon=1e-6, name=f"ln2_{block_id}")(x)
    
    return x


class TransformerModel(BaseModel):     # 继承 BaseModel
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None,
                 d_model=256, num_heads=4, ff_dim=512, dropout_rate=0.2, num_layers=4, p=2,
                 l2_reg=1e-5, use_gating=True, use_pos_encoding=True, class_weights=None, loss_type='focal_loss',
                 predict_type=PredictType.CLASSIFY
                 ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.p = p
        self.l2_reg = l2_reg
        self.use_gating = use_gating
        self.use_pos_encoding = use_pos_encoding
        if fn is not None:
            self.model = type(self).load(fn, custom_objects={"CrossNet": CrossNet, "Time2Vec": Time2Vec})
            self.history = LossHistory(predict_type=predict_type, test_x=test_x, test_y=test_y)
            return
        super().__init__(x=x, y=y, test_x=test_x, test_y=test_y,
                         loss_type=loss_type, class_weights=class_weights,
                         predict_type=predict_type)

    def _build(self, shape):
        inputs = Input(shape=shape)
        x_seq = LayerNormalization(epsilon=1e-6)(inputs)
        x_seq = Dense(self.d_model, kernel_regularizer=l2(self.l2_reg))(x_seq)
        if self.use_pos_encoding:
            seq_length = shape[0]
            positions = tf.range(start=0, limit=seq_length, delta=1, dtype=tf.float32)
            positions = tf.reshape(positions, (1, seq_length, 1))
            t2v_layer = Time2Vec(self.d_model, kernel_regularizer=l2(self.l2_reg))
            time_embedding = t2v_layer(positions)
            x_seq = x_seq + time_embedding
        for i in range(self.num_layers):
            x_seq = transformer_encoder_block(
                x_seq,
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
                block_id=i,
                use_gating=self.use_gating
            )
        x = GlobalAveragePooling1D()(x_seq)
        x = Dense(128, activation='gelu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='gelu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate / 2)(x)
        outputs = self.build_output_head(x, self.predict_type)   # 共用输出头
        self.model = Model(inputs=inputs, outputs=outputs)