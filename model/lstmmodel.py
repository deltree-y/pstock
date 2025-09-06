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
from keras.utils import class_weight
from utils.const_def import BASE, NUM_CLASSES

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

class LSTMModel():
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, p=2):
        if fn != None:
            self.load(fn)
            self.model.summary()
        elif x is not None and y is not None:
            self.p = p
            self.x,self.y1,self.y2 = x.astype(float),y[:,0].astype(int),y[:,1].astype(int)
            self.test_x, self.test_y1, self.test_y2 = test_x.astype(float), test_y[:,0].astype(int), test_y[:,1].astype(int)
            self.history = LossHistory()
            logging.debug(f"input shape is - <{x.shape}>")
            self.create_model(x[0].shape)
            self.model.summary()
        else:
            logging.error("LSTMModel init fail, no fn or x/y input!")
            exit()

    def create_model(self, shape):
            inputs = Input(shape)
            
            # 第一层 Bidirectional LSTM - 减少正则化强度
            x = Bidirectional(LSTM(self.p*32, return_sequences=True, kernel_regularizer=l2(1e-5)))(inputs)
            x = LayerNormalization()(x)
            x = Dropout(0.2)(x)  # 降低dropout率

            # 第二层 Bidirectional LSTM
            x = Bidirectional(LSTM(self.p*16, return_sequences=False, kernel_regularizer=l2(1e-5)))(x)
            x = LayerNormalization()(x)
            x = Dropout(0.3)(x)  # 降低dropout率

            # 减少Dense层复杂度，防止过拟合
            shared = Dense(self.p*64, activation='relu', kernel_regularizer=l2(1e-5))(x)  # 从256减少到64
            shared = Dropout(0.3)(shared)

            shared = Dense(self.p*32, activation='relu', kernel_regularizer=l2(1e-5))(shared)  # 从64减少到32
            shared = Dropout(0.2)(shared)

            # 输出层
            out1 = Dense(self.p*16, activation='relu', kernel_regularizer=l2(1e-5))(shared)  # 从32减少到16
            out1 = Dropout(0.1)(out1)  # 最后一层使用更低的dropout
            out1 = Dense(NUM_CLASSES, activation='softmax', name='output1')(out1)

            self.model = Model(inputs=inputs, outputs=out1)

    def train(self, epochs=100, batch_size=32):
        model_path = os.path.join(BASE, "model", f"stocks_{epochs}_best.h5")
        
        # 模型检查点 - 保存最佳模型
        mc = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch'
        )
        
        # 学习率调度器 - 当验证损失停止改善时降低学习率
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=8,  # 增加patience，避免过早降低学习率
            min_lr=1e-7,  # 设置最小学习率
            verbose=1
        )
        
        # 早停机制 - 防止过拟合
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # 15个epoch没有改善就停止
            restore_best_weights=True,
            verbose=1
        )

        # 改进的优化器配置
        self.model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),  # 添加梯度裁剪，使用适中的学习率
            loss={'output1': 'sparse_categorical_crossentropy'},
            metrics={'output1': 'accuracy'},
            weighted_metrics=[]
        )
        
        start_time = datetime.now()
        
        # 计算类别权重以处理类别不平衡问题
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(self.y1),
            y=self.y1
        )
        class_weight_dict = dict(enumerate(class_weights))
        logging.info(f"Class weights computed: {class_weight_dict}")
        
        # 添加所有callback
        callbacks = [mc, self.history, lr_scheduler, early_stopping]
        
        self.model.fit(
            x=self.x,
            y={'output1': self.y1},
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y1), 
            validation_freq=1, 
            callbacks=callbacks,
            epochs=epochs, 
            shuffle=True, 
            verbose=1,  # 改为1以便观察训练过程
            class_weight=class_weight_dict  # 添加类别权重
        )
        
        spend_time = datetime.now() - start_time
        return "\n total spend:%.2f(h)/%.1f(m), %.1f(s)/epoc, %.2f(h)/10k"\
              %(spend_time.seconds/3600, spend_time.seconds/60, spend_time.seconds/epochs, 10000*(spend_time.seconds/3600)/epochs)

    def save(self, filename):
        try:
            self.model.save(filename)
            logging.info(f"model file saved -[{filename}]")
        except:
            logging.error("model file save failed!")
            exit()

    def load(self, filename):
        try:
            print("loading model file -[%s]..."%(filename),end="",flush=True)
            self.model = load_model(filename)
            print("complete!")
        except:
            logging.error("model file load failed!")
            exit()

    def plot(self):
        train_loss = self.history.get_loss()
        train_t1_accu, _ = self.history.get_accu()
        val_loss = self.history.get_val_loss()
        val_t1_accu, val_t2_accu = self.history.get_val()
        epochs = range(1, len(train_loss)+1)
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.plot(epochs, val_t1_accu, label='Validation T1 Accuracy')
        plt.plot(epochs, train_t1_accu, label='Train T1 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve and Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def create_model_sequential(self, shape):
        self.model_sequential = Sequential([
            Bidirectional(LSTM(1024, input_shape=shape, kernel_regularizer=l2(0.01),\
                               return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(256, kernel_regularizer=l2(0.01), \
                               return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), \
                               return_sequences=False)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(2)
        ])
