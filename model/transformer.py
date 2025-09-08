# coding=utf-8
import os, sys, logging
import numpy as np
from datetime import datetime
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, MultiHeadAttention, Add, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from model.history import LossHistory

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate):
    # Self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Feed Forward Network
    ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(1e-4))(x)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output_proj = Dense(x.shape[-1], activation=None)(ff_output)
    x = Add()([x, ff_output_proj])
    x = LayerNormalization()(x)
    return x

class TransformerModel():
    def __init__(self, x=None, y=None, test_x=None, test_y=None, 
                 d_model=256, num_heads=4, ff_dim=512, dropout_rate=0.2, num_layers=4, p=2):
        self.x, self.y1, self.test_x, self.test_y1 = x, y, test_x, test_y
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.p = p
        self.history = LossHistory()
        logging.info(f"Transformer input shape: {x.shape}, output shape: {y.shape}")
        self.create_model(x[0].shape)
        self.model.summary()

    def create_model(self, shape):
        inputs = Input(shape)
        x = LayerNormalization()(inputs)
        for _ in range(self.num_layers):
            x = transformer_encoder_block(x, self.d_model, self.num_heads, self.ff_dim, self.dropout_rate)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(self.dropout_rate)(x)
        out = Dense(1, activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=out)
        


    def train(self, epochs=80, batch_size=512, learning_rate=0.002, patience=30):
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=int(patience/4), min_lr=1e-6, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
                           loss='mse',
                           metrics=['mae'])
        start_time = datetime.now()
        self.history.set_para(epochs, start_time)
        self.model.fit(
            x=self.x,
            y=self.y1,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y1),
            epochs=epochs,
            callbacks=[self.history, lr_scheduler, early_stopping],
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