# codeing: utf-8
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, LayerNormalization, Activation
from keras.layers import Conv1D, BatchNormalization

class ResidualTCN(Model):
    def __init__(self, input_shape, nb_filters=64, kernel_size=3, nb_stacks=1, dilations=None,
                dropout_rate=0.1, return_sequences=False):
        super(ResidualTCN, self).__init__()
        
        # 默认膨胀率设置
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]
            
        self.input_shape_val = input_shape
        self.residual_blocks = []
        
        # 创建残差块
        for stack_i in range(nb_stacks):
            for dilation in dilations:
                self.residual_blocks.append(
                    ResidualBlock(nb_filters, kernel_size, dilation, dropout_rate)
                )
                
        # 输出层
        self.final_norm = LayerNormalization()
        self.final_dropout = Dropout(dropout_rate)
        self.final_activation = Activation('relu')
        self.output_layer = Dense(1, activation='linear')  # 回归任务使用linear激活
        
    def call(self, inputs, training=None):
        x = inputs
        for block in self.residual_blocks:
            x = block(x, training=training)
            
        # 仅取最后一个时间步的输出（用于回归任务）
        x = x[:, -1, :]
        
        # 最终处理
        x = self.final_norm(x)
        if training:
            x = self.final_dropout(x)
        x = self.final_activation(x)
        return self.output_layer(x)
    
    def build_graph(self):
        input_layer = Input(shape=self.input_shape_val)
        return Model(inputs=[input_layer], outputs=self.call(input_layer))


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, nb_filters, kernel_size, dilation_rate, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding='causal')
        self.norm1 = LayerNormalization()
        self.activation1 = Activation('relu')
        self.dropout1 = Dropout(dropout_rate)
        
        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                           dilation_rate=dilation_rate, padding='causal')
        self.norm2 = LayerNormalization()
        self.activation2 = Activation('relu')
        self.dropout2 = Dropout(dropout_rate)
        
        # 如果输入和输出形状不同，添加1x1卷积进行映射
        self.use_projection = False
        self.projection = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.use_projection = True
            self.projection = Conv1D(filters=self.conv1.filters, kernel_size=1, padding='same')
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # 跳跃连接的原始输入
        residual = inputs
        
        # 第一个卷积块
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation1(x)
        if training:
            x = self.dropout1(x)
            
        # 第二个卷积块
        x = self.conv2(x)
        x = self.norm2(x)
        
        # 处理残差连接
        if self.use_projection:
            residual = self.projection(residual)
            
        # 添加残差连接
        x = Add()([x, residual])
        x = self.activation2(x)
        if training:
            x = self.dropout2(x)
            
        return x