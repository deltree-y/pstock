# codeing: utf-8
import os, sys, logging
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import layers
from datetime import datetime
from tcn import TCN
from model.losses import focal_loss
from utils.const_def import NUM_CLASSES
from model.history import LossHistory
from model.utils import WarmUpCosineDecayScheduler

class TCNModel():
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, p=2,
                 nb_filters=64, kernel_size=8, nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32], dropout_rate=0.3,
                 class_weights=None, loss_type=None
                 ):
        if fn != None:
            self.load(fn)
            self.model.summary()
        elif x is not None and y is not None:
            self.p = p
            self.nb_filters = nb_filters
            self.kernel_size = kernel_size
            self.dropout_rate = dropout_rate
            self.nb_stacks = nb_stacks
            self.dilations = dilations or [1, 2, 4, 8, 16, 32]
            self.class_weights = class_weights
            self.loss_type = loss_type
            self.learning_rate_status = "init"

            self.x, self.y = x.astype(float), y.astype(int)
            self.test_x, self.test_y = test_x.astype(float), test_y.astype(int)
            self.history = None
            logging.info(f"TCN input shape: {x.shape}, output shape: {y.shape}")

            self.create_model(x[0].shape)
            self.model.summary()
        else:
            logging.error("RCNModel init fail, no fn or x/y input!")
            exit()

    # 在TCN层中添加归一化层
    def improved_residual_block(self, x, dilation_rate, filters, kernel_size=8):
        # 残差连接
        residual = layers.Conv1D(filters=filters, kernel_size=1, padding='same',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        
        # 第一卷积层
        conv = layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                        dilation_rate=dilation_rate, padding='causal',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        conv = layers.LayerNormalization()(conv)  # 使用Layer Normalization
        conv = layers.Activation('relu')(conv)
        conv = layers.SpatialDropout1D(self.dropout_rate)(conv)
        
        # 第二卷积层
        conv = layers.Conv1D(filters=filters, kernel_size=kernel_size,
                        dilation_rate=dilation_rate, padding='causal',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-5))(conv)
        conv = layers.LayerNormalization()(conv)  # 使用Layer Normalization
        conv = layers.Activation('relu')(conv)
        conv = layers.SpatialDropout1D(self.dropout_rate)(conv)
        
        # 合并
        out = layers.Add()([residual, conv])
        return out

    # 针对TCNModel的优化建议
    def create_model(self, shape):
        """构建TCN模型"""
        input_shape = self.x.shape[1:]
        inputs = Input(shape=input_shape)
        x = inputs
        
        # 多层TCN残差块，扩张卷积逐层增加
        for stack in range(self.nb_stacks):
            for dilation in self.dilations:
                x = self.improved_residual_block(x, 
                                       dilation_rate=dilation,
                                       filters=self.nb_filters,
                                       kernel_size=self.kernel_size)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        
        # 全连接层
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        # 多分类输出
        outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output1')(x)
        self.model = Model(inputs=inputs, outputs=outputs)

    def train(self, tx, ty, epochs=120, batch_size=256, learning_rate=0.001, patience=20):
        self.x = tx.astype('float32') if tx is not None else self.x
        self.y = ty.astype(int) if ty is not None else self.y

        # 多分类损失
        if self.loss_type == 'focal_loss':
            loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        else:
            loss_fn = 'sparse_categorical_crossentropy'

        # Huber损失函数，对异常值更鲁棒
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss={'output1': loss_fn},
            metrics={'output1': 'accuracy'}
        )

        # 添加学习率调度和早停
        warmup_steps, hold_steps = int(0.2 * epochs), int(0.3 * epochs)  
        lr_scheduler = WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate,
            total_steps=epochs,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        # 开始训练
        self.history = LossHistory()
        start_time = datetime.now()
        self.history.set_para(epochs, start_time)
        self.model.fit(
            x=self.x, y=self.y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_x, self.test_y),
            callbacks=[self.history, lr_scheduler, early_stopping],
            shuffle=True,
            class_weight=self.class_weights,
            verbose=0
        )
        
        spend_time = datetime.now() - start_time
        return "\n total spend:%.2f(h)/%.1f(m), %.1f(s)/epoc, %.2f(h)/10k" \
              % (spend_time.seconds/3600, spend_time.seconds/60, 
                 spend_time.seconds/epochs, 10000*(spend_time.seconds/3600)/epochs)


    def save(self, filename):
        try:
            self.model.save(filename)
            logging.info(f"model file saved: {filename}")
        except Exception as e:
            logging.error(f"model file save failed! {e}")
            exit()

    def load(self, filename):
        from keras.models import load_model
        try:
            print(f"loading model file: {filename} ...", end="", flush=True)
            self.model = load_model(filename, custom_objects={'TCN': TCN})
            print("complete!")
        except Exception as e:
            logging.error(f"model file load failed! {e}")
            exit()

