# coding=utf-8
import os, sys, logging
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.metrics import confusion_matrix
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from datasets.stockinfo import StockInfo
from dataset import StockDataset
from predicproc.predict import Predict
from model.residual_lstm import ResidualLSTMModel
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, BASE_DIR, MODEL_DIR, NUM_CLASSES
from utils.utils import setup_logging, print_ratio
from predicproc.analyze import plot_confusion
from model.utils import auto_adjust_class_weights, confusion_based_weights

if __name__ == "__main__":
    setup_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # ================== 数据集准备 ==================
    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = []
    related_stock_list = ALL_CODE_LIST

    # 注意：train_size 设大容易使验证集过少，这里保持 0.9 但你可以回退到 0.85 观察稳定性
    ds = StockDataset(ts_code=primary_stock_code,
                      idx_code_list=index_code_list,
                      rel_code_list=related_stock_list,
                      si=si,
                      start_date='20200104',
                      end_date='20250903',
                      train_size=0.9,
                      if_use_all_features=False)

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y

    # 仅使用 y 的第一列作为回归目标（T1低值变化率 *100 后的结果）
    ty1 = ty[:, 0]#.astype(float)
    vy1 = vy[:, 0]#.astype(float)


    # ================== 训练参数 ==================
    epochs = 120
    batch_size = 1024
    learning_rate = 0.00006
    patience = 50
    cls_weights = dict(enumerate([0.1854275092976686, 1.042750929367235, 1.2682527881028205, 1.226394052044119, 0.8714498141287835, 1.405724907059373]))

    # 模型结构配置
    depth = 6          # 残差块数
    base_units = 32    # 每方向 LSTM 基础单元（最终 BiLSTM 输出通道=2*base_units*p）
    p = 2              # 放大系数
    dropout_rate = 0.3
    use_se = True

    tm = ResidualLSTMModel(
        x=tx,
        y=ty1,
        test_x=vx,
        test_y=vy1,
        p=p,
        depth=depth,
        base_units=base_units,
        dropout_rate=dropout_rate,
        use_se=use_se,
        se_ratio=8,
        l2_reg=1e-5,
        class_weights=cls_weights,
        #loss_fn=loss_fn
    )
    logging.info(f"\nbins1: {ds.bins1.prop_bins}\nbins2: {ds.bins2.prop_bins}")
    print_ratio(ty1, "ty1")

    logging.info(f"Start training: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
    logging.info(f"tx shape: {tx.shape}, ty1 shape: {ty1.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
    train_ret = tm.train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience
    )
    logging.info(train_ret)

    # ================== 评估 ==================
    y_pred = tm.model.predict(vx, batch_size=2048).reshape(-1)
    # ================== 保存模型 ==================
    save_path = os.path.join(BASE_DIR, MODEL_DIR, f"{primary_stock_code}_ResidualLSTM_ep{epochs}_bs{batch_size}_p{p}_d{depth}.h5")
    tm.save(save_path)

    # ================== 指定日期预测 ==================
    t_list = ['20250829', '20250901', '20250902', '20250903']
    for t0 in t_list:
        print(f"Predict for T0[{t0}]")
        data, bp = ds.get_predictable_dataset_by_date(t0)
        pred_scaled = tm.model.predict(data)
        logging.info(f"Predict scaled result: {pred_scaled}")
        Predict(pred_scaled, bp, ds.bins1, ds.bins2).print_predict_result()
        print()

    if True:
        y_pred = tm.model.predict(vx)
        y_pred_label = np.argmax(y_pred, axis=1)
        print_ratio(y_pred_label, "y_pred_label")
        auto_adjust_class_weights(y_pred_label, NUM_CLASSES)
        confusion_based_weights(vy1, y_pred_label, NUM_CLASSES)
        print_ratio(vy1, "vy1")
        auto_adjust_class_weights(vy1, NUM_CLASSES)

        cm = plot_confusion(vy1, y_pred_label, num_classes=NUM_CLASSES)
        if False:
            cm = confusion_matrix(vy[:, 0], y_pred_label)
            plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.colorbar()
            plt.show()

