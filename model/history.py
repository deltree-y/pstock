import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from keras.callbacks import Callback
from utils.const_def import ACCU_RATE_THRESHOLD


class LossHistory(Callback):
    def __init__(self, predict_type=None, test_x=None, test_y=None):
        super(LossHistory, self).__init__()
        self.predict_type = predict_type
        self.test_x = test_x
        self.test_y = test_y

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.t1_accu, self.t2_accu = [], []
        self.val_accu, self.val_t1_accu, self.val_t2_accu = [], [], []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if self.predict_type.is_regress():# 回归任务
            if self.test_x is not None and self.test_y is not None:
                acc = ""
                acc_str,val_acc_str = "--"
                y_pred_val = self.model.predict(self.test_x, verbose=0).reshape(-1)
                val_accu = np.mean(np.abs(y_pred_val - self.test_y) <= ACCU_RATE_THRESHOLD)
                val_acc_str = f"{val_accu*100:.2f}"
                self.val_accu.append(val_accu)
            else:
                acc_str, val_acc_str = "--/--", "--/--"
        else:# 分类任务
            acc = logs.get('accuracy')
            val_accu = logs.get('val_accuracy')
            acc_str = f"{acc*100:.2f}" if acc is not None else "--"
            val_acc_str = f"{val_accu*100:.2f}" if val_accu is not None else "--"
            self.val_accu.append(val_accu)

        #计算损失变化率
        loss_diff_ratio = 100*(self.val_losses[-1] - self.val_losses[-2]) / self.val_losses[-2] if epoch > 0 else 0
        
        #计算每轮时间和预计完成时间
        spend_time = datetime.now() - self.start_time
        speed = spend_time.seconds / (epoch + 1)
        min_remaining = speed * (self.epoch - epoch - 1) / 60
        finished_time = (timedelta(seconds=(speed * (self.epoch - epoch - 1))) + datetime.now()).strftime('%H:%M')

        # 只输出回归损失
        if logs:
            print(f"\n{epoch + 1}/{self.epoch}: "
                  f"t:[{logs.get('loss'):.4f}/{acc_str}], "
                  f"v:[{logs.get('val_loss'):.4f}/{val_acc_str}]({loss_diff_ratio:+.2f}%),",
                  f"lr({self.model.learning_rate_status}):{tf.keras.backend.get_value(self.model.optimizer.lr):.6f}",
                  f"{speed:.1f}s/ep, ed:{finished_time}({min_remaining/60:.1f}h)", 
                  end="", flush=True)
                    
            loss_improve_str, val_improve_str = None, None
            if epoch > 0 and logs.get('val_loss') < min(self.val_losses[:-1]):  #val_loss更小时增加打印项目
                loss_improve_str = f"[{min(self.val_losses[:-1])-logs.get('val_loss'):.5f}]"
                loss_improve_ratio_str = f"[{100*(min(self.val_losses[:-1])-logs.get('val_loss'))/min(self.val_losses[:-1]):.2f}%]"
            if epoch > 0 and val_accu > max(self.val_accu[:-1]):    #val_accuracy更大时增加打印项目
                val_improve_str = f"[{(val_accu-max(self.val_accu[:-1]))*100:.2f}]"
            if loss_improve_str is not None or val_improve_str is not None:
                print("  <-- ", end="")
                if loss_improve_str is not None:
                    print(f"l{loss_improve_ratio_str}", end=" ")
                if val_improve_str is not None:
                    print(f"v{val_improve_str}", end=" ")
                print("", end="", flush=True)
        
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