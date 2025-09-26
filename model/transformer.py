# coding=utf-8
import os, sys, logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, MultiHeadAttention, Add, Flatten, Lambda
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.utils import compute_class_weight
from model.utils import WarmUpCosineDecayScheduler
from utils.const_def import NUM_CLASSES
from model.history import LossHistory
from model.losses import focal_loss

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

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
    """
    优化建议2: 增强的Transformer编码器块，支持门控机制选项
    优化建议3: 增加L2正则化和更多dropout位置
    """
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


class TransformerModel():
    def __init__(self, x=None, y=None, test_x=None, test_y=None, 
                 d_model=256, num_heads=4, ff_dim=512, dropout_rate=0.2, num_layers=4, p=2,
                 l2_reg=1e-5, use_gating=False, use_pos_encoding=True, class_weights=None, loss_type='focal_loss'
                 ):
        """
        优化建议2: 增加更多模型配置选项
        优化建议3: 添加L2正则化参数
        优化建议7: 支持选择不同的损失函数
        
        参数:
            x, y: 训练数据和标签
            test_x, test_y: 测试数据和标签
            d_model: 模型维度
            num_heads: 注意力头数量
            ff_dim: 前馈网络维度
            dropout_rate: Dropout率
            num_layers: Transformer层数
            p: 放大系数
            l2_reg: L2正则化系数
            use_gating: 是否使用门控机制
            use_pos_encoding: 是否使用位置编码
            class_weights: 类别权重
            loss_type: 损失函数类型 ('focal_loss', 'cross_entropy', 'weighted_cross_entropy')
        """
        self.x = x.astype('float32')
        self.y = y.astype(int)  # 多分类标签（分箱后的类别编号）
        self.test_x = test_x.astype('float32') if test_x is not None else None
        self.test_y = test_y.astype(int) if test_y is not None else None
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.p = p
        self.l2_reg = l2_reg
        self.learning_rate_status = "init"
        
        self.use_gating = use_gating
        self.use_pos_encoding = use_pos_encoding
        self.loss_type = loss_type
        
        self.class_weight_dict = class_weights

        self.history = LossHistory()
        logging.info(f"Transformer input shape: {x.shape}, output shape: {y.shape}")
        self.create_model(x[0].shape)
        self.model.summary()

    def create_model(self, shape):
        """
        构建改进的Transformer模型
        优化建议1: 添加位置编码
        优化建议2: 更灵活的网络结构配置
        优化建议3: 增强正则化
        """
        inputs = Input(shape)
        
        # 初始归一化和投影
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = Dense(self.d_model, kernel_regularizer=l2(self.l2_reg))(x)
        
        # 优化建议1: 添加位置编码
        if self.use_pos_encoding:
            seq_length = inputs.shape[1]
            pos_encoding = positional_encoding(seq_length, self.d_model)
            x = x + pos_encoding
        
        # 构建多层Transformer编码器
        for i in range(self.num_layers):
            x = transformer_encoder_block(
                x,
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                l2_reg=self.l2_reg,
                block_id=i,
                use_gating=self.use_gating
            )
        
        # 序列展平并进行最终分类
        x = Flatten()(x)
        x = Dense(128, activation='gelu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='gelu', kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate/2)(x)  # 输出层前降低dropout以稳定训练
        
        # 输出层
        temperature = 1.25
        x_last = Dense(NUM_CLASSES, name='logits')(x)
        outputs = Lambda(lambda x: tf.nn.softmax(x / temperature), name='output1')(x_last)

        self.model = Model(inputs=inputs, outputs=outputs)
        

    def train(self, tx, ty, epochs=80, batch_size=512, learning_rate=0.002, weight_decay=1e-5, patience=30):
        """
        训练模型
        优化建议4: 使用AdamW优化器并添加weight_decay参数
        优化建议7: 支持不同的损失函数
        """
        self.x = tx.astype('float32') if tx is not None else self.x
        self.y = ty.astype(int) if ty is not None else self.y

        optimizer = optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5)#AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=0.5)

        # 优化建议7: 支持不同的损失函数
        if self.loss_type == 'focal_loss':
            loss = focal_loss(gamma=2.0, alpha=0.25)
        elif self.loss_type == 'weighted_cross_entropy':
            loss = 'sparse_categorical_crossentropy'
        else:  # 默认交叉熵
            loss = 'sparse_categorical_crossentropy'

        self.model.compile(
            optimizer=optimizer,
            loss={'output1': loss},
            metrics={'output1': 'accuracy'}
        )        

        # 添加学习率调度和早停
        warmup_steps, hold_steps = int(0.2 * epochs), int(0.2 * epochs)
        lr_scheduler = WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate,
            total_steps=epochs,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps  # 5%的步数保持不变
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        start_time = datetime.now()
        self.history.set_para(epochs, start_time)
        self.model.fit(
            x=self.x, y=self.y,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y),
            epochs=epochs,
            callbacks=[self.history, lr_scheduler, early_stopping],
            class_weight=self.class_weight_dict,
            shuffle=True,
            verbose=0
        )
        return "Train finished."

    def save(self, filename):
        try:
            self.model.save(filename)
            logging.info(f"\nmodel file saved -[{filename}]")
        except Exception as e:
            logging.error(f"\nmodel file save failed! {e}")
            exit()

    def load(self, filename):
        from keras.models import load_model
        try:
            print(f"\nloading model file -[{filename}]...", end="", flush=True)
            self.model = load_model(filename)
            print("complete!")
        except Exception as e:
            logging.error(f"\nmodel file load failed! {e}")
            exit()