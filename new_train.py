# coding=utf-8
import os, sys, logging
import numpy as np
#from pathlib import Path
#o_path = os.getcwd()
#sys.path.append(o_path)
#sys.path.append(str(Path(__file__).resolve().parents[0]))

from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.residual_lstm import ResidualLSTMModel
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, BASE_DIR, MODEL_DIR, NUM_CLASSES
from utils.utils import setup_logging, print_ratio
from predicproc.analyze import plot_confusion_by_model, print_predict_result
from model.utils import  get_hard_samples, get_sample_weights

if __name__ == "__main__":
    setup_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # ================== 数据集准备 ==================
    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    t_list = ['20250829', '20250901', '20250902', '20250903']
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
    n_repeat = 3
    epochs = 120
    batch_size = 1024
    learning_rate = 0.00005
    patience = 50
    cls_weights = dict(enumerate([0.5, 1.1, 1.1, 1.1, 1.1, 1.1]))
    #cls_weights = dict(enumerate([0.1854275092976686, 1.042750929367235, 1.2682527881028205, 1.226394052044119, 0.8714498141287835, 1.405724907059373]))

    # 模型结构配置
    depth = 6          # 残差块数
    base_units = 32    # 每方向 LSTM 基础单元（最终 BiLSTM 输出通道=2*base_units*p）
    p = 2              # 放大系数
    dropout_rate = 0.3
    use_se = True

    res_lstm = ResidualLSTMModel(
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

    # 1. 正常训练
    logging.info(f"1. 正常训练: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
    logging.info(f"tx shape: {tx.shape}, ty1 shape: {ty1.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
    train_ret = res_lstm.train(tx=tx, ty=ty1, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
    print_predict_result(t_list, ds, res_lstm)
    #plot_confusion_by_model(res_lstm, vx, vy1, num_classes=NUM_CLASSES, title=f"1. 正常训练 Confusion Matrix")
    
    # ================== 重点训练 ==================
    # 2. 获取 hard 样本
    logging.info("2. 获取 hard 样本")
    hard_x, hard_y = get_hard_samples(tx, ty1, res_lstm.model, threshold=0.25)
    aug_x, aug_y = ds.time_series_augmentation_4x(hard_x, hard_y, noise_level=0.01)

    # 3. 增权/增强/多次训练任选其一或多种组合
    if True:
        # 3.1 增权训练(先找出预测置信度低的样本，给它们更高的权重)
        hard_mask = np.max(res_lstm.model.predict(tx), axis=1) < 0.25
        res_lstm.class_weight_dict = dict(enumerate(get_sample_weights(ty1, hard_mask)))
        logging.info(f"3.1 增权训练: tx shape: {tx.shape}, ty1 shape: {ty1.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
        train_ret = res_lstm.train(tx=tx, ty=ty1, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        print_predict_result(t_list, ds, res_lstm)
        #plot_confusion_by_model(res_lstm, vx, vy1, num_classes=NUM_CLASSES, title=f"3.1 增权训练: Confusion Matrix")

    if False:
        # 3.2 数据增强训练
        logging.info(f"3.2 数据增强训练: tx shape: {aug_x.shape}, ty1 shape: {aug_y.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
        res_lstm.train(tx=aug_x, ty=aug_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        print_predict_result(t_list, ds, res_lstm)
        #plot_confusion_by_model(res_lstm, vx, vy1, num_classes=NUM_CLASSES, title=f"3.2 数据增强训练: Confusion Matrix")

    if True:
        # 3.3 多次训练（可以和增权、增强结合）
        logging.info(f"3.3 多次训练: tx shape: {hard_x.shape}, ty1 shape: {hard_y.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
        for _ in range(n_repeat):
            res_lstm.train(tx=hard_x, ty=hard_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        print_predict_result(t_list, ds, res_lstm)
        #plot_confusion_by_model(res_lstm, vx, vy1, num_classes=NUM_CLASSES, title=f"3.3 多次训练: Confusion Matrix")

    # 或者将增强后的 hard 样本拼回主数据集再训练
    if True:
        # 3.4 将增强后的 hard 样本拼回主数据集再训练
        final_train_x = np.concatenate([tx, aug_x])
        final_train_y = np.concatenate([ty1, aug_y])
        logging.info(f"3.4 将增强后的 hard 样本拼回主数据集再训练: tx shape: {final_train_x.shape}, ty1 shape: {final_train_y.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
        res_lstm.train(tx=final_train_x, ty=final_train_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        print_predict_result(t_list, ds, res_lstm)
        plot_confusion_by_model(res_lstm, vx, vy1, num_classes=NUM_CLASSES, title=f"3.4 将增强后的 hard 样本拼回主数据集再训练: Confusion Matrix")

    # ================== 评估 ==================
    #y_pred = res_lstm.model.predict(vx, batch_size=2048).reshape(-1)

    # ================== 保存模型 ==================
    save_path = os.path.join(BASE_DIR, MODEL_DIR, f"{primary_stock_code}_ResidualLSTM_ep{epochs}_bs{batch_size}_p{p}_d{depth}.h5")
    res_lstm.save(save_path)
