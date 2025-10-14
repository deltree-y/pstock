import os, sys, logging
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Dropout,
    Add, GlobalAveragePooling1D, Dense, Multiply, LayerNormalization
)
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from datetime import datetime
from utils.const_def import NUM_CLASSES, IS_PRINT_MODEL_SUMMARY
from utils.utils import PredictType
from model.losses import get_loss
from model.history import LossHistory
from model.utils import WarmUpCosineDecayScheduler

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

class Conv1DResModel:
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, 
                 filters=64, kernel_size=5, depth=4, 
                 dropout_rate=0.2, l2_reg=1e-5, use_se=True, se_ratio=8,
                 class_weights=None, loss_type=None, 
                 predict_type=PredictType.CLASSIFY):
        if fn is not None:
            self.load(fn)
            self.model.summary() if IS_PRINT_MODEL_SUMMARY else None
            return

        self.x = x.astype('float32') if x is not None else None
        self.y = y.astype(int) if y is not None else None
        self.test_x = test_x.astype('float32') if test_x is not None else None
        self.test_y = test_y.astype(int) if test_y is not None else None
        self.filters = filters
        self.kernel_size = kernel_size
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.predict_type = predict_type
        self.learning_rate_status = "init"
        self.history = LossHistory()
        logging.info(f"Conv1DResModel: input shape={self.x.shape if self.x is not None else None}, y shape={self.y.shape if self.y is not None else None}")

        if x is not None:
            self.create_model(self.x.shape[1:])
            self.model.summary() if IS_PRINT_MODEL_SUMMARY else None

    def create_model(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp
        # 多层残差Conv1D
        for i in range(self.depth):
            x = residual_conv_block(
                x, filters=self.filters, kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate, l2_reg=self.l2_reg,
                use_se=self.use_se, se_ratio=self.se_ratio, block_id=i
            )
        x = GlobalAveragePooling1D(name="gap")(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg), name="fc1")(x)
        x = Dropout(self.dropout_rate, name="fc1_drop")(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg), name="fc2")(x)
        x = Dropout(self.dropout_rate, name="fc2_drop")(x)
        if self.predict_type.is_classify():
            out = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
        elif self.predict_type.is_binary():
            out = Dense(1, activation='sigmoid', name='output')(x)
        else:
            raise ValueError("Unsupported predict_type for classification model.")
        self.model = Model(inputs=inp, outputs=out)

    def train(self, tx, ty, epochs=100, batch_size=256, learning_rate=0.001, patience=20):
        self.x = tx.astype('float32') if tx is not None else self.x
        self.y = ty.astype(int) if ty is not None else self.y
        loss_fn = get_loss(self.loss_type, self.predict_type)
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss={'output': loss_fn},
            metrics={'output': 'accuracy'}
        )
        # 学习率调度和早停
        warmup_steps, hold_steps = int(0.2 * epochs), int(0.2 * epochs)
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
        start_time = datetime.now()
        self.history.set_para(epochs, start_time)
        self.model.fit(
            x=self.x, y=self.y,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y),
            validation_freq=1,
            callbacks=[self.history, lr_scheduler, early_stopping],
            epochs=epochs,
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
            logging.info(f"\nmodel file saved -[{filename}]")
        except Exception as e:
            logging.error(f"\nmodel file save failed! {e}")
            exit()

    def load(self, filename):
        try:
            print(f"\nloading model file -[{filename}]...", end="", flush=True)
            self.model = load_model(filename)
            print("complete!")
        except Exception as e:
            logging.error(f"\nmodel file load failed! {e}")
            exit()