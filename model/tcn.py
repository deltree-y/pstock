# codeing: utf-8
import os, sys, logging
import numpy as np
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from datetime import datetime
from model.history import LossHistory
from tcn import TCN

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
            self.nb_stacks = nb_stacks
            self.dilations = dilations
            self.dropout_rate = dropout_rate

            self.x, self.y = x.astype(float), y.astype(float)
            self.test_x, self.test_y = test_x.astype(float), test_y.astype(float)
            self.history = LossHistory()
            logging.info(f"TCN input shape: {x.shape}, output shape: {y.shape}")

            self.create_model(x[0].shape)
            self.model.summary()
        else:
            logging.error("RCNModel init fail, no fn or x/y input!")
            exit()

    def create_model(self, shape):
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

    def train(self, epochs=120, batch_size=256, learning_rate=0.001, patience=20):
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=int(patience/4) if patience>=50 else int(patience/2), min_lr=1e-5, verbose=1)

        callbacks = [self.history, early_stopping, lr_scheduler]
        start_time = datetime.now()
        self.history.set_para(epochs, start_time)

        self.model.fit(
            self.x, self.y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_x, self.test_y),
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
            workers=4,  # 多线程加速
            use_multiprocessing=True
        )


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