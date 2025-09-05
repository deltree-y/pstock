# codeing: utf-8
import os, sys, logging
import numpy as np
from keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from datetime import datetime
from pathlib import Path
from history import LossHistory
from utils.const_def import BASE, PREDICT_CNT

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

class StockModel():
    def __init__(self, input_shape=None, filename=None, is_need_summary=False):
        if input_shape != None:
            self.input_shape = input_shape
            logging.debug(f"input shape is - <{self.input_shape}>")
            self.model = None
            self.history = LossHistory()
            self.spend_time = 0
            self.create_model()
        if filename != None:
            self.load(filename)
        if is_need_summary:
            self.model.summary()

    def create_model(self):
        inputs = Input(self.input_shape)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(64, (2, 2), strides=(1, 1), #padding='same',
                   activation='relu', name='conv0')(inputs)
        X = BatchNormalization(axis=2, name='bn0')(X)
        X = Dropout(0.5)(X)

        # MAXPOOL
        X = MaxPooling2D((2, 2), #strides=(1, 1), #padding='same',
                         name='max_pool')(X)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(128, (2, 2), strides=(1, 1), #padding='same',
                   activation='relu',  name='conv1')(X)
        X = BatchNormalization(axis=2, name='bn1')(X)
        X = Dropout(0.5)(X)


        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(256, (4, 4), strides=(1, 1), #padding='same',
                   activation='relu',  name='conv1.5')(X)
        X = BatchNormalization(axis=2, name='bn1.5')(X)
        X = Dropout(0.5)(X)

        # CONV -> BN -> RELU Block applied to X
        X = Conv2D(1024, (4, 4), strides=(1, 1), #padding='same',
                   activation='relu', name='conv2')(X)
        X = BatchNormalization(axis=2, name='bn2')(X)
        X = Dropout(0.5)(X)

        # CONV
        X = Conv2D(64, (2, 2), strides=(1, 1), #padding='same',
                   activation='relu', name='conv3')(X)
        X = BatchNormalization(axis=2, name='bn3')(X)
        X = Dropout(0.5)(X)

        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        X = Dense(1024, activation='relu', name='fc1')(X)
        X = Dropout(0.5)(X)

        X = Dense(512, activation='relu', name='fc2')(X)
        X = Dropout(0.5)(X)
        output1 = Dense(self.input_shape[2], activation='softmax', name='stock1')(X)
        output2 = Dense(self.input_shape[2], activation='softmax', name='stock2')(X)
        output3 = Dense(self.input_shape[2], activation='softmax', name='stock3')(X)

        self.model = Model(inputs=inputs, outputs=[output1, output2, output3], name='BaseModel')

    def loss_train(self, X, Y, x_val=None, y_val=None, batch_size=64, epochs=1):
        in_Y = Y

        mc=ModelCheckpoint(
            BASE + "\\model\\" + str(PREDICT_CNT) +"_stocks_"+str(epochs)+'_best.h5',
            monitor='loss', #'val_accuracy', #'val_loss',
            verbose=2,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch'
        )
        lrs=ReduceLROnPlateau(
            monitor='loss', 
            factor=0.1,
            patience=100,
            verbose=1,
            mode='auto',
            min_delta=1e-4,
            cooldown=1,
            min_lr=0.001
        )

        self.model.compile(optimizer='adam',#(learning_rate=0.05), 
                            loss={"stock1":"categorical_crossentropy","stock2":"categorical_crossentropy","stock3":"categorical_crossentropy"}, 
                            metrics={'stock1': ['accuracy'], 'stock2': ['accuracy'], 'stock3': ['accuracy']})
        start_time = datetime.now()
        history=self.model.fit(x=X, 
                                y={
                                    'stock1': in_Y[:, 0, :],
                                    'stock2': in_Y[:, 1, :],
                                    'stock3': in_Y[:, 2, :]
                                },                                
                                batch_size=batch_size, 
                                #validation_split=0.15,
                                validation_data=(x_val, {
                                    'stock1': y_val[:, 0, :],
                                    'stock2': y_val[:, 1, :],
                                    'stock3': y_val[:, 2, :]
                                }), 
                                validation_freq= 1, 
                                callbacks=[mc, self.history],
                                epochs=epochs, 
                                shuffle=True, 
                                verbose=0)
        self.spend_time = datetime.now() - start_time
        print("\nINFO: total spend:%.2f(h)/%.1f(m), %.1f(s)/epoc, %.2f(h)/10k"\
              %(self.spend_time.seconds/3600,self.spend_time.seconds/60,self.spend_time.seconds/epochs,10000*(self.spend_time.seconds/3600)/epochs))
        #with open('log_sgd_big_32.txt','w') as f:
        #    f.write("time [%s]\n"%(datetime.now()))
        #    f.write(str(history.history))

    def pred_data(self, in_x):
        #ret = self.model.predict(np.expand_dims(in_x, axis=0), verbose=0)
        ret = self.model(np.expand_dims(in_x, axis=0))
        ret = np.array(ret)
        #ret = ((ret[0]).tolist())
        #accu = max(ret)
        #print("DEBUG: accu is [%s]"%str(accu))
        #return ret, accu
        return ret

    def save(self, filename):
        try:
            self.model.save(filename)
            print("INFO: model file saved -[%s]"%(filename))
        except:
            print("ERROR: model file save failed!")
            sys.exit()

    def load(self, filename):
        try:
            print("INFO: loading model file -[%s]..."%(filename),end="",flush=True)
            self.model = load_model(filename)
            print("complete!")
        except:
            print("ERROR: model file load failed!")
            sys.exit()


