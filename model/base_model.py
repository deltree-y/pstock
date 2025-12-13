import logging
from abc import ABC, abstractmethod
from datetime import datetime

from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Dense, Lambda

from model.history import LossHistory
from model.utils import WarmUpCosineDecayScheduler
from model.losses import get_loss
from utils.utils import PredictType
from utils.const_def import IS_PRINT_MODEL_SUMMARY, NUM_CLASSES


class BaseModel(ABC):
    """
    各模型共享的通用基类：数据缓存、损失/指标选择、训练循环、保存/加载等。
    子类只需实现 `_build(self, input_shape)` 完成具体网络搭建。
    """

    def __init__(
        self,
        x=None,
        y=None,
        test_x=None,
        test_y=None,
        loss_type=None,
        class_weights=None,
        predict_type: PredictType = PredictType.CLASSIFY,
    ):
        self.x = x.astype("float32") if x is not None else None
        self.y = y if y is not None else None
        self.test_x = test_x.astype("float32") if test_x is not None else None
        self.test_y = test_y if test_y is not None else None

        self.loss_type = loss_type
        self.class_weights = class_weights
        self.predict_type = predict_type
        self.learning_rate_status = "init"

        # 训练历史
        self.history = LossHistory(
            predict_type=self.predict_type,
            test_x=self.test_x,
            test_y=self.test_y,
        )

        # 留给子类设置的 Keras 模型
        self.model = None
        if self.x is not None:
            self._build(self.x.shape[1:])
            self.model.summary() if IS_PRINT_MODEL_SUMMARY else None


    # ---------- 子类需实现 ----------
    @abstractmethod
    def _build(self, input_shape):
        """子类负责根据 input_shape 搭建 self.model"""
        raise NotImplementedError

    # ---------- 通用输出头定义 ----------
    def build_output_head(self, x_last, predict_type:PredictType, name_prefix="output", regress_scale=5.0):
        if predict_type.is_classify():
            return Dense(NUM_CLASSES, activation='softmax', name=name_prefix)(x_last)
        elif predict_type.is_binary():
            return Dense(1, activation='sigmoid', name=name_prefix)(x_last)
        elif predict_type.is_regress():
            logits = Dense(1, activation='tanh', name=f"{name_prefix}_logits")(x_last)
            return Lambda(lambda t: regress_scale * t, name=name_prefix)(logits)
        else:
            raise ValueError(f"Unsupported predict_type: {predict_type}")

    # ---------- 训练/编译通用逻辑 ----------
    def _compile(self, learning_rate):
        loss_fn = get_loss(self.loss_type, self.predict_type)
        metrics = {"output": ["mae", "mse"]} if self.predict_type.is_regress() else {"output": "accuracy"}
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=0.5),
            loss={"output": loss_fn},
            metrics=metrics,
        )

    def _callbacks(self, epochs, patience, learning_rate):
        warmup_steps, hold_steps = int(0.1 * epochs), int(0.1 * epochs) # 预热和保持各10%的训练周期
        monitor_metric = "val_mae" if self.predict_type.is_regress() else "val_loss"    # 回归监控验证 MAE，分类监控验证 Loss

        lr_scheduler = WarmUpCosineDecayScheduler(
            learning_rate_base=learning_rate,
            total_steps=epochs,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps,
        )
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
        return [self.history, lr_scheduler, early_stopping]

    def train(
        self,
        tx=None,
        ty=None,
        epochs=100,
        batch_size=256,
        learning_rate=0.001,
        patience=20,
    ):
        # 允许外部传入新数据
        self.x = tx.astype("float32") if tx is not None else self.x
        self.y = ty if ty is not None else self.y

        if self.model is None:
            raise ValueError("model is not built yet.")

        self._compile(learning_rate)
        callbacks = self._callbacks(epochs, patience, learning_rate)

        start_time = datetime.now()
        self.history.set_para(epochs, start_time)

        self.model.fit(
            x=self.x,
            y=self.y,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y),
            validation_freq=1,
            callbacks=callbacks,
            epochs=epochs,
            shuffle=True,
            class_weight=self.class_weights,
            verbose=0,
        )
        spend = datetime.now() - start_time
        return "\n total spend:%.2f(h)/%.1f(m), %.1f(s)/epoc, %.2f(h)/10k" % (
            spend.seconds / 3600,
            spend.seconds / 60,
            spend.seconds / epochs,
            10000 * (spend.seconds / 3600) / epochs,
        )

    # ---------- 通用 IO ----------
    def save(self, filename):
        try:
            self.model.save(filename)
            logging.info(f"\nmodel file saved -[{filename}]")
        except Exception as e:
            logging.error("\nmodel file save failed!")
            raise

    @classmethod
    def load(cls, filename, custom_objects=None):
        """
        若子类有自定义层，需要在 custom_objects 传入。
        调用方式：ChildModel.load(...) 后再包一层子类实例化逻辑。
        """
        try:
            print(f"\nloading model file -[{filename}]...", end="", flush=True)
            m = load_model(filename, custom_objects=custom_objects)
            print("done.")
            m.summary() if IS_PRINT_MODEL_SUMMARY else None
            return m
        except Exception as e:
            logging.error(f"model file load failed! {e}")
            raise