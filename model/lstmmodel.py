# codeing: utf-8
import os, sys, logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from keras.models import Sequential, Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, LayerNormalization
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
from model.history import LossHistory
from sklearn.utils.class_weight import compute_class_weight
from model.utils import WarmUpCosineDecayScheduler
from utils.const_def import NUM_CLASSES, IS_PRINT_MODEL_SUMMARY

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

class LSTMModel():
    def __init__(self,
                 x=None, y=None,
                 test_x=None,test_y=None,
                 fn=None, p=2,
                 depth=3, base_units=32,use_se=True, se_ratio=8,
                 dropout_rate=0.2, l2_reg=1e-5,
                 loss_fn=None, class_weights=None, loss_type=None
                 ):
        if fn is not None:
            self.load(fn)
            self.model.summary()
            return
        
        self.p = p
        self.x = x.astype('float32')
        self.y = y.astype(int)  # 分类任务 y 应为整数类别
        self.test_x = test_x.astype('float32') if test_x is not None else None
        self.test_y = test_y.astype(int) if test_y is not None else None
        self.loss_type = loss_type
        self.learning_rate_status = "init"

        self.history = LossHistory()
        self.depth = depth
        self.base_units = base_units
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.l2_reg = l2_reg
        self.loss_fn = loss_fn  # 保存传入的自定义损失（可为 None）
        self.class_weight_dict = class_weights

        self.history = LossHistory()
        self.create_model_mini(x[0].shape)
        self.model.summary() if IS_PRINT_MODEL_SUMMARY else None
        logging.info(f"LSTM Mini Model: input shape={self.x.shape}, y shape={self.y.shape}")

    def create_model_mini(self, shape):
        inputs = Input(shape)
        x = LSTM(128, return_sequences=False)(inputs)
        x = Dense(32, activation='relu')(x)
        out1 = Dense(NUM_CLASSES, activation='softmax', name='output1')(x)
        self.model = Model(inputs=inputs, outputs=out1)

    def train(self, tx, ty, epochs=100, batch_size=32, learning_rate=0.001, patience=30):
        self.x = tx.astype('float32') if tx is not None else self.x
        self.y = ty.astype(int) if ty is not None else self.y
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss={'output1': 'sparse_categorical_crossentropy'},
            metrics={'output1': 'accuracy'}
        )        
        
        # 添加学习率调度和早停
        warmup_steps, hold_steps = int(0.4 * epochs), int(0.1 * epochs)
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
        
        # 添加所有callback
        callbacks = [self.history, lr_scheduler, early_stopping]
        
        self.model.fit(
            x=self.x, y=self.y,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y),
            validation_freq=1,
            callbacks=callbacks,
            epochs=epochs, 
            shuffle=True, 
            verbose=0  # 改为0以减少输出
        )
        
        spend_time = datetime.now() - start_time
        return "\n total spend:%.2f(h)/%.1f(m), %.1f(s)/epoc, %.2f(h)/10k"\
              %(spend_time.seconds/3600, spend_time.seconds/60, spend_time.seconds/epochs, 10000*(spend_time.seconds/3600)/epochs)


    def save(self, filename):
        try:
            self.model.save(filename)
            logging.info(f"\nmodel file saved -[{filename}]")
        except:
            logging.error("\nmodel file save failed!")
            exit()

    def load(self, filename):
        try:
            print("\nloading model file -[%s]..."%(filename),end="",flush=True)
            self.model = load_model(filename)
            print("complete!")
        except:
            logging.error("\nmodel file load failed!")
            exit()
