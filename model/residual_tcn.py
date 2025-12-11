# coding: utf-8
import os, sys, logging
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Add, LayerNormalization, Activation, Conv1D, Lambda
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from datetime import datetime
from utils.utils import PredictType
from utils.const_def import NUM_CLASSES, IS_PRINT_MODEL_SUMMARY
from model.losses import binary_focal_loss, focal_loss, get_loss
from model.history import LossHistory
from model.utils import WarmUpCosineDecayScheduler

class ResidualBlock(tf.keras.layers.Layer):
    """
    单层TCN残差块：两次扩张卷积->归一化->激活->Dropout->Add残差
    支持因果卷积与通道数自动投影
    """
    def __init__(self, nb_filters, kernel_size, dilation_rate, block_id, dropout_rate=0.1, l2_reg=1e-5, causal=True, **kwargs):
        #K.clear_session()
        #super(ResidualBlock, self).__init__(name=f"residual_block_{block_id}", **kwargs)
        super().__init__(**kwargs)
        self.causal = causal
        self.block_id = block_id
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding='causal' if causal else 'same',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                            name=f"c{block_id}")
        self.norm1 = LayerNormalization(name=f"ln1_{block_id}")
        self.activation1 = Activation('relu', name=f"act1_{block_id}")
        self.dropout1 = Dropout(dropout_rate, name=f"drop1_{block_id}")

        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding='causal' if causal else 'same',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                            name=f"conv2_{block_id}")
        self.norm2 = LayerNormalization(name=f"ln2_{block_id}")
        self.activation2 = Activation('relu', name=f"act2_{block_id}")
        self.dropout2 = Dropout(dropout_rate, name=f"drop2_{block_id}")

        self.use_projection = False
        self.projection = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "nb_filters": self.conv1.filters,
            "kernel_size": self.conv1.kernel_size[0],
            "dilation_rate": self.conv1.dilation_rate[0],
            "block_id": self.block_id,     # 新增
            "dropout_rate": self.dropout1.rate,
            "l2_reg": self.conv1.kernel_regularizer.l2 if self.conv1.kernel_regularizer else 0.0,
            "causal": self.causal,
        })
        return config
    
    def build(self, input_shape):
        # 若输入通道和输出通道不同，用1x1卷积投影
        if input_shape[-1] != self.conv1.filters:
            self.use_projection = True
            self.projection = Conv1D(filters=self.conv1.filters, kernel_size=1, padding='same', name=f"proj_{self.block_id}")
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation1(x)
        if training:
            x = self.dropout1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        if training:
            x = self.dropout2(x)
        if self.use_projection:
            residual = self.projection(residual)
        out = Add(name=f"add_{self.block_id}")([x, residual])
        return out, x  # 返回主路径和skip分支

class ResidualTCNModel:
    """
    主流残差TCN模型，支持因果卷积/多stack堆叠/skip分支加和
    """
    def __init__(self, x=None, y=None, test_x=None, test_y=None, fn=None, p=2,
                 nb_filters=64, kernel_size=8, nb_stacks=1, dilations=None, dropout_rate=0.1,
                 l2_reg=1e-5, causal=True,
                 class_weights=None, loss_type=None,
                 predict_type=PredictType.CLASSIFY
                 ):
        if fn is not None:
            self.load(fn)
            self.model.summary() if IS_PRINT_MODEL_SUMMARY else None
            return
        self.p = p
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations or [1, 2, 4, 8, 16, 32]
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.causal = causal
        self.x = x.astype(float)
        self.y = y
        self.test_x = test_x.astype(float) if test_x is not None else None
        self.test_y = test_y if test_y is not None else None
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.predict_type = predict_type

        self.learning_rate_status = "init"
        self.history = LossHistory(predict_type=self.predict_type, test_x=self.test_x, test_y=self.test_y)
        logging.info(f"ResidualTCN input shape: {self.x.shape}, output shape: {self.y.shape}")
        self.create_model(self.x.shape[1:])

    def create_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = inputs
        skip_connections = []

        # 多stack堆叠，每stack多层dilation
        for stack in range(self.nb_stacks):
            for dilation in self.dilations:
                block_id = f"{stack}_{dilation}"
                block = ResidualBlock(self.nb_filters, self.kernel_size, dilation, block_id,
                                      self.dropout_rate, self.l2_reg, causal=self.causal)
                x, skip = block(x)
                skip_connections.append(skip)

        # 所有skip分支加和
        if len(skip_connections) > 1:
            x = Add()(skip_connections)
        else:
            x = skip_connections[0]

        # 仅取最后一个时间步的输出
        x = x[:, -1, :]
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Activation('relu')(x)

        # 输出层
        #temperature = 1.25
        #x_last = Dense(NUM_CLASSES, name='logits')(x)
        #outputs = Lambda(lambda x: tf.nn.softmax(x / temperature), name='output')(x_last)
        if self.predict_type.is_classify():
            outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
        elif self.predict_type.is_binary():
            outputs = Dense(1, activation='sigmoid', name='output')(x)
        elif self.predict_type.is_regress():
            logits = Dense(1, activation='tanh', name='logits')(x)
            outputs = Lambda(lambda t: 5.0 * t, name='output')(logits)
        else:
            raise ValueError("Unsupported predict_type for classification model.")            
        self.model = Model(inputs=inputs, outputs=outputs)

    def train(self, tx, ty, epochs=120, batch_size=256, learning_rate=0.001, patience=20):
        self.x = tx.astype('float32') if tx is not None else self.x
        self.y = ty if ty is not None else self.y
        metrics = {'output': ['mae','mse']} if self.predict_type.is_regress() else {'output': 'accuracy'}
        monitor_metric = 'val_mae' if self.predict_type.is_regress() else 'val_loss'

        loss_fn = get_loss(self.loss_type, self.predict_type)
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss={'output': loss_fn},
            metrics=metrics
        )
        warmup_steps, hold_steps = int(0.2 * epochs), int(0.2 * epochs)
        lr_scheduler = WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate,
            total_steps=epochs,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps
        )
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
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
            self.model.save(filename, save_format='tf')
            logging.info(f"model file saved: {filename}")
        except Exception as e:
            logging.error(f"model file({filename}) save failed : {e}")
            exit()

    def load(self, filename):
        from keras.models import load_model
        try:
            from model.residual_tcn import ResidualBlock  # 关键：导入自定义层
            print(f"loading model file: {filename} ...", end="", flush=True)
            self.model = load_model(filename, custom_objects={"ResidualBlock": ResidualBlock})
            print("complete!")
        except Exception as e:
            logging.error(f"model file load failed! {e}")
            exit()