# codeing: utf-8
import os, sys, logging
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout, Dense, GlobalAveragePooling1D, LayerNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import layers
from keras.losses import Huber
from datetime import datetime
from tcn import TCN
from model.history import LossHistory
from model.utils import WarmUpCosineDecayScheduler, direction_weighted_mse

class TCNModel():
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, p=2,
                 nb_filters=64, kernel_size=8, nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32], dropout_rate=0.3):
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

            self.x, self.y = x.astype(float), y.astype(float)
            self.test_x, self.test_y = test_x.astype(float), test_y.astype(float)
            self.history = None
            logging.info(f"TCN input shape: {x.shape}, output shape: {y.shape}")

            self.create_model(x[0].shape)
            self.model.summary()
        else:
            logging.error("RCNModel init fail, no fn or x/y input!")
            exit()

    def residual_block(self, x, dilation_rate, filters, kernel_size=8):
        """
        残差块的实现，包含因果卷积、归一化和非线性激活
        """
        # 第一个卷积层路径
        residual = layers.Conv1D(filters=filters,
                                kernel_size=1,
                                padding='same')(x)
        
        # 主卷积路径
        conv = layers.Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           dilation_rate=dilation_rate,
                           padding='causal')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.SpatialDropout1D(self.dropout_rate)(conv)
        
        conv = layers.Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           dilation_rate=dilation_rate,
                           padding='causal')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.SpatialDropout1D(self.dropout_rate)(conv)
        
        # 合并残差连接
        out = layers.Add()([residual, conv])
        return out

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
                #x = self.residual_block(x, 
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
        outputs = layers.Dense(1)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)

    def train(self, epochs=120, batch_size=256, learning_rate=0.001, patience=20):
        self.history = LossHistory()
        self.history.set_para(epochs, datetime.now())

        # Huber损失函数，对异常值更鲁棒
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss=Huber(delta=0.1),  # 使用Huber损失而不是MSE
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
        #优化前的学习率调整
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience/2,
            min_lr=1e-6,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        callbacks = [self.history, lr_scheduler, early_stopping]

        # 开始训练
        start_time = datetime.now()
        self.model.fit(
            self.x, self.y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_x, self.test_y),
            callbacks=callbacks,
            shuffle=True,
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

    #更复杂的模型
    def create_model_history(self, shape):
        from keras.regularizers import l2
        inputs = Input(shape)
        x = TCN(
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=self.nb_stacks,
            dilations=self.dilations,
            dropout_rate=self.dropout_rate,
            return_sequences=False
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.5)(x)
        out = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=out)

