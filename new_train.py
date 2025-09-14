# coding=utf-8
import os, sys, logging
import numpy as np
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))

from datasets.stockinfo import StockInfo
from dataset import StockDataset
from predicproc.predict import RegPredict
from model.residual_lstm import ResidualLSTMModel
from model.losses import mse_with_variance_push
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, BASE_DIR, MODEL_DIR
from utils.utils import setup_logging, plot_regression_result, plot_error_distribution

if __name__ == "__main__":
    setup_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # ================== 数据集准备 ==================
    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = []
    related_stock_list = ALL_CODE_LIST

    # 注意：train_size 设大容易使验证集过少，这里保持 0.9 但你可以回退到 0.85 观察稳定性
    ds = StockDataset(primary_stock_code,
                      index_code_list,
                      related_stock_list,
                      si,
                      start_date='20180101',
                      end_date='20250903',
                      train_size=0.9)

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y

    # 仅使用 y 的第一列作为回归目标（T1低值变化率 *100 后的结果）
    ty_reg = ty[:, 0].astype(float)
    vy_reg = vy[:, 0].astype(float)

    #mean_y, std_y = np.mean(ty_reg), np.std(ty_reg)
    #ty_reg_scaled = (ty_reg - mean_y) / std_y
    #vy_reg_scaled = (vy_reg - mean_y) / std_y

    # 简单噪声增强（如不需要，直接改成 x_aug, y_aug = tx, ty_reg_scaled）
    #x_aug, y_aug = ds.time_series_augmentation(tx, ty_reg_scaled, noise_level=0.01)
    #x_aug, y_aug = tx, ty_reg_scaled   # 若想关闭增强，取消注释本行

    #logging.info(f"Train X shape: {x_aug.shape}, y shape: {y_aug.shape}")
    #logging.info(f"Valid X shape: {vx.shape}, y shape: {vy_reg_scaled.shape}")

    # ================== 训练参数 ==================
    epochs = 120
    batch_size = 1024
    learning_rate = 1e-3
    patience = 25

    # 模型结构配置
    depth = 4          # 残差块数
    base_units = 48    # 每方向 LSTM 基础单元（最终 BiLSTM 输出通道=2*base_units*p）
    p = 2              # 放大系数
    dropout_rate = 0.25
    use_se = True

    #losses配置
    var_floor_ratio = 0.5   # 要求预测 std >= 50% 的真实 std
    penalty_weight = 0.05   # 方差惩罚权重（可调）

    # 构造 loss_fn
    loss_fn = mse_with_variance_push(var_floor_ratio=var_floor_ratio, penalty_weight=penalty_weight)

    tm = ResidualLSTMModel(
        x=tx,
        y=ty_reg,
        test_x=vx,
        test_y=vy_reg,
        p=p,
        depth=depth,
        base_units=base_units,
        dropout_rate=dropout_rate,
        use_se=use_se,
        se_ratio=8,
        l2_reg=1e-5#,
        #loss_fn=loss_fn
    )

    logging.info(f"Start training: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
    train_ret = tm.train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    logging.info(train_ret)

    # ================== 评估 ==================
    y_pred_scaled = tm.model.predict(vx, batch_size=2048).reshape(-1)
    y_pred = y_pred_scaled# * std_y + mean_y

    mae = np.mean(np.abs(y_pred - vy_reg))
    rmse = np.sqrt(np.mean((y_pred - vy_reg) ** 2))
    logging.info(f"回归评估: MAE={mae:.5f}, RMSE={rmse:.5f}")

    # 可视化
    plot_regression_result(vy_reg, y_pred, title="Validation Real vs Pred")
    plot_error_distribution(vy_reg, y_pred, title="Validation Error Distribution")

    # ================== 保存模型 ==================
    save_path = os.path.join(BASE_DIR, MODEL_DIR,
                             f"{primary_stock_code}_ResidualLSTM_ep{epochs}_bs{batch_size}_p{p}_d{depth}.h5")
    tm.save(save_path)

    # ================== 指定日期预测 ==================
    t_list = ['20250829', '20250901', '20250902', '20250903']
    for t0 in t_list:
        print(f"Predict for T0[{t0}]")
        data, bp = ds.get_predictable_dataset_by_date(t0)
        pred_scaled = tm.model.predict(data)
        #RegPredict(pred_scaled, bp, std_y, mean_y).print_predict_result()
        RegPredict(pred_scaled, bp).print_predict_result()
        print()