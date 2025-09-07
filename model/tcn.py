# codeing: utf-8
import os, sys, logging
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Dense, Add, Lambda
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from model.history import LossHistory

class TCNModel():
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, p=2):
        if fn != None:
            self.load(fn)
            self.model.summary()
        elif x is not None and y is not None:
            self.x, self.y = x, y
            self.test_x, self.test_y = test_x, test_y
            self.p = p
            self.history = LossHistory()
            self.create_model(x.shape[1:], num_filters=32*p, kernel_size=3, num_blocks=3)
        else:
            logging.error("RCNModel init fail, no fn or x/y input!")
            exit()

    def residual_block(self, x, filters, kernel_size, dilation_rate, dropout_rate):
        # Causal Conv1D
        conv = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(dropout_rate)(conv)
        # 1x1 Conv to match dimensions
        res = Conv1D(filters, 1, padding='same')(x)
        out = Add()([conv, res])
        return out

    def create_model(self, input_shape, num_filters=32, kernel_size=3, num_blocks=3, dropout_rate=0.2):
        inputs = Input(shape=input_shape)
        x = inputs
        for i in range(num_blocks):
            x = self.residual_block(
                x, filters=num_filters, kernel_size=kernel_size,
                dilation_rate=2**i, dropout_rate=dropout_rate
            )
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Lambda(lambda t: t[:, -1, :])(x)  # 只用最后一步特征
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=out)

    def train(self, epochs=100, batch_size=64, learning_rate=0.001, patience=30):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss='mse',
            metrics=['mae']
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.9, patience=int(patience/5), min_lr=1e-6, verbose=0
        )
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
        )
        callbacks = [self.history, lr_scheduler, early_stopping]

        self.model.fit(
            x=self.x, y=self.y,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y),
            validation_freq=1, 
            epochs=epochs,
            callbacks=callbacks,
            shuffle=True,
            verbose=0
        )

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        from keras.models import load_model
        self.model = load_model(filename)