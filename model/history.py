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
        self.val_t1_accu.append(logs.get('val_accuracy'))

        loss_diff = self.val_losses[-1] - self.val_losses[-2] if epoch > 0 else self.val_losses[-1]
        loss_diff_ratio = 100*(self.val_losses[-2] - self.val_losses[-1]) / self.val_losses[-2] if epoch > 0 else 0
        spend_time = datetime.now() - self.start_time
        speed = spend_time.seconds / (epoch + 1)
        min_remaining = speed * (self.epoch - epoch - 1) / 60
        finished_time = (timedelta(seconds=(speed * (self.epoch - epoch - 1))) + datetime.now()).strftime('%H:%M')
        # 只输出回归损失
        if logs:
            print(f"\n{epoch + 1}/{self.epoch}: "
                  f"t:[{logs.get('loss'):.4f}/{logs.get('accuracy')*100:.2f}], "
                  #f"v:[{logs.get('val_loss'):.4f}/{logs.get('val_accuracy')*100:.2f}]({loss_diff:+.4f}),",
                  f"v:[{logs.get('val_loss'):.4f}/{logs.get('val_accuracy')*100:.2f}]({loss_diff_ratio:+.2f}%),",
                  f"lr({self.model.learning_rate_status}):{tf.keras.backend.get_value(self.model.optimizer.lr):.6f}",
                  f"{speed:.1f}s/ep, ed:{finished_time}({min_remaining/60:.1f}h)", end="", flush=True)
                    
            loss_improve_str, val_improve_str = None, None
            if epoch > 0 and logs.get('val_loss') < min(self.val_losses[:-1]):
                loss_improve_str = f"[{min(self.val_losses[:-1])-logs.get('val_loss'):.5f}]"
                loss_improve_ratio_str = f"[{100*(min(self.val_losses[:-1])-logs.get('val_loss'))/min(self.val_losses[:-1]):.2f}%]"
            if epoch > 0 and logs.get('val_accuracy') > max(self.val_t1_accu[:-1]):
                val_improve_str = f"[{(logs.get('val_accuracy')-max(self.val_t1_accu[:-1]))*100:.2f}]"
            if loss_improve_str is not None or val_improve_str is not None:
                print("  <-- ", end="")
                if loss_improve_str is not None:
                    #print(f"l{loss_improve_str}", end=" ")
                    print(f"l{loss_improve_ratio_str}", end=" ")
                if val_improve_str is not None:
                    print(f"v{val_improve_str}", end=" ")
                print("", end="", flush=True)

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