import tensorflow as tf
from datetime import datetime, timedelta
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

        loss_diff = self.val_losses[-1] - self.val_losses[-2] if epoch > 0 else self.val_losses[-1]
        spend_time = datetime.now() - self.start_time
        speed = spend_time.seconds / (epoch + 1)
        min_remaining = speed * (self.epoch - epoch - 1) / 60
        finished_time = (timedelta(seconds=(speed * (self.epoch - epoch - 1))) + datetime.now()).strftime('%H:%M')
        # 只输出回归损失
        if logs:
            print(f"\n{epoch + 1}/{self.epoch}: "
                  f"l/a=[{logs.get('loss'):.4f}/{logs.get('accuracy')*100:.2f}], "
                  f"val l/a=[{logs.get('val_loss'):.4f}/{logs.get('val_accuracy')*100:.2f}]({loss_diff:+.4f}),",
                  f"lr({self.model.learning_rate_status}):{tf.keras.backend.get_value(self.model.optimizer.lr):.8f}",
                  f"{speed:.1f}s/epo,ed:{finished_time}({min_remaining/60:.1f}h)", end="", flush=True)
                    
            if epoch > 0 and logs.get('val_loss') < min(self.val_losses[:-1]):
                print(f" <-- [{min(self.val_losses[:-1])-logs.get('val_loss'):.5f}]", end="", flush=True)


        ### 以下所有行多分类时使用 ###
        if False:
            self.t1_accu.append(logs.get('accuracy')*100)
            self.val_t1_accu.append(logs.get('val_accuracy')*100)
            # 改进的日志记录 - 每个epoch都记录，便于监控训练过程
            if logs:
                print(f"\nEpoch {epoch + 1}: "
                            f"loss={logs.get('loss'):.4f}, "
                            f"val_loss={logs.get('val_loss'):.4f}, "
                            f"acc={logs.get('accuracy')*100:.2f}%, "
                            f"val_acc={logs.get('val_accuracy')*100:.2f}%",end="")
                
                # 添加验证损失改善检测
                if epoch > 0 and logs.get('val_loss') < min(self.val_losses[:-1]):
                    print(f"  <-- Validation loss improved!", end="")
        
        return super().on_epoch_end(epoch, logs)

    def get_last_loss(self):
        if self.losses and self.val_losses:
            return self.losses[-1], self.val_losses[-1]
        return None, None

    def get_best_val_loss(self):
        val_loss = self.model.history.history.get('val_loss', [])
        return min(val_loss)
    
    def get_best_val(self):
        #return max(self.t1_accu), max(self.t2_accu)
        return max(self.val_t1_accu)
   
    def get_history(self):
        for key in self.model.history.history.keys():
            print(key)
            print("DEBUG: %s"%str(self.model.history.history[key]))
            print()
        return self.model.history
    
    def set_para(self, epoch, start_time):
        self.epoch = epoch
        self.start_time = start_time