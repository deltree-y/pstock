import logging
from keras.callbacks import Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.t1_accu = []
        self.t2_accu = []
        self.val_losses = []
        self.val_t1_accu = []
        self.val_t2_accu = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.t1_accu.append(logs.get('accuracy')*100)
        self.val_t1_accu.append(logs.get('val_accuracy')*100)
        
        # 改进的日志记录 - 每个epoch都记录，便于监控训练过程
        if logs:
            logging.info(f"Epoch {epoch + 1}: "
                        f"loss={logs.get('loss'):.4f}, "
                        f"val_loss={logs.get('val_loss'):.4f}, "
                        f"acc={logs.get('accuracy')*100:.2f}%, "
                        f"val_acc={logs.get('val_accuracy')*100:.2f}%")
            
            # 添加验证损失改善检测
            if epoch > 0 and logs.get('val_loss') < min(self.val_losses[:-1]):
                logging.info(f"  --> Validation loss improved!")
        
        return super().on_epoch_end(epoch, logs)

    def on_batch_end(self, batch, logs={}):
        pass
        #print("DEBUG: %s"%str(logs))
        #self.val_losses.append(logs.get('val_loss'))

    def get_last_loss(self):
        val_loss = self.model.history.history.get('val_loss', [])
        return self.losses[-1], val_loss[-1]
    
    def get_best_val_loss(self):
        val_loss = self.model.history.history.get('val_loss', [])
        return min(val_loss)
    
    def get_loss(self):
        #return self.model.history.history.get('loss', [])
        return self.losses
    
    def get_accu(self):
        return self.t1_accu, self.t2_accu
    
    def get_val_loss(self):
        #return self.model.history.history.get('val_loss', [])
        return self.val_losses
    
    def get_val(self):
        #return self.model.history.history.get('val_output1_accuracy', []), self.model.history.history.get('val_output2_accuracy', [])
        return self.val_t1_accu, self.val_t2_accu

    def get_best_val(self):
        #return max(self.t1_accu), max(self.t2_accu)
        return max(self.val_t1_accu)
   
    def get_history(self):
        for key in self.model.history.history.keys():
            print(key)
            print("DEBUG: %s"%str(self.model.history.history[key]))
            print()
        return self.model.history

    def get_last_val(self):
        val1 = self.model.history.history.get('val_output1_accuracy', [])
        val2 = self.model.history.history.get('val_output2_accuracy', [])
        return round(val1[-1],5), round(val2[-1],5)