# coding=utf-8
import os, sys, time, argparse, datetime, logging
import numpy as np
import pandas as pd
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from predicproc.predict import Predict, RegPredict
from model.lstmmodel import LSTMModel
from model.tcn import TCNModel
from model.transformer import TransformerModel
from model.residual_tcn import ResidualTCN
from model.residual_lstm import ResidualLSTMModel
from sklearn.metrics import confusion_matrix
from utils.tk import TOKEN
from utils.const_def import NUM_CLASSES, BANK_CODE_LIST, ALL_CODE_LIST
from utils.const_def import BASE_DIR, MODEL_DIR
from utils.utils import feature_importance_analysis, setup_logging, select_features_by_tree_importance, auto_select_features, select_features_by_stat_corr, plot_regression_result, plot_error_distribution
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_vol_norm", action="store_true", help="Use window-level volatility normalization targets from StockDataset")
    ap.add_argument("--start_date", type=str, default='20180101')
    ap.add_argument("--end_date", type=str, default='20250903')
    ap.add_argument("--train_size", type=float, default=0.9)
    return ap.parse_args()

if __name__ == "__main__":
    setup_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = parse_args()
    use_vol_norm = args.use_vol_norm
    if_print_detail = False
    
    # 优化后的训练参数 - 使用更多的epoch和更好的batch size
    epo_list = [100]  # 增加epochs，早停会自动停止
    p_list = [2]
    batch_size_list = [1024]  # 增加batch size以提高训练稳定性
    learning_rate = 0.001  # 使用更高的初始学习率
    patience = 20  # 提高早停的耐心值，允许更多epoch的波动
    dropout_rate = 0.3
    
    #以下2个为TCN参数
    nb_filters = 128  #64~256,越大越复杂
    kernel_size = 12   #3~16,越大越复杂
    
    #以下3个为Transformer参数
    num_layers = 4  #4-8层,越高越复杂
    d_model=128
    ff_dim=256


    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = []#'000001.SH']#, '399001.SZ', '399006.SZ']  #上证指数,深证成指,创业板指
    related_stock_list = ALL_CODE_LIST  # 关联股票列表
    # 改善数据集配置 - 使用更好的train/validation分割比例
    ds = StockDataset(primary_stock_code, index_code_list, related_stock_list, si, start_date='20180101',end_date='20250903', train_size=0.9)  # 90%/10%分割提供更多验证数据

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    ### 只用T1 low的涨跌幅为回归目标 ###
    ty_reg = ty[:, 0].astype(float)
    vy_reg = vy[:, 0].astype(float)

    mean_y, std_y = np.mean(ty_reg), np.std(ty_reg)
    ty_reg_scaled = (ty_reg - mean_y) / std_y
    vy_reg_scaled = (vy_reg - mean_y) / std_y
    ty_reg_scaled_default = (ty_reg - mean_y) / std_y
    vy_reg_scaled_default = (vy_reg - mean_y) / std_y if vy_reg.size>0 else np.array([])

    # ------------------ VOL NORM handling ------------------
    if use_vol_norm:
        # Check dataset has precomputed vol-norm targets
        has_vol_fields = all(hasattr(ds, k) for k in ("train_y_vol_norm_scaled", "train_volnorm_mean", "train_volnorm_std", "windowed_train_vol_pct"))
        if not has_vol_fields or ds.train_y_vol_norm_scaled is None or ds.train_y_vol_norm_scaled.size == 0:
            logging.warning("Dataset does not provide vol-normalized targets. Falling back to raw z-scored targets.")
            use_vol_norm = False
        else:
            logging.info("Using dataset-provided volatility-normalized targets (train_y_vol_norm_scaled).")
            # use dataset's scaled targets for training
            ty_reg_scaled = ds.train_y_vol_norm_scaled
            vy_reg_scaled = getattr(ds, "test_y_vol_norm_scaled", np.array([]))
            # store for inverse transform later
            volnorm_mean = ds.train_volnorm_mean
            volnorm_std = ds.train_volnorm_std
            train_vol_pct = getattr(ds, "windowed_train_vol_pct", np.array([]))
            test_vol_pct = getattr(ds, "windowed_test_vol_pct", np.array([]))

            # Align lengths between tx and ty_reg_scaled if necessary
            if len(ty_reg_scaled) != tx.shape[0]:
                minlen = min(len(ty_reg_scaled), tx.shape[0])
                logging.warning(f"[ALIGN] vol-norm train y / x length mismatch (y:{len(ty_reg_scaled)} vs x:{tx.shape[0]}), trimming to {minlen}")
                ty_reg_scaled = ty_reg_scaled[:minlen]
                tx = tx[:minlen]
            if vx is not None and vx.size>0 and vy_reg_scaled is not None and vy_reg_scaled.size>0 and len(vy_reg_scaled) != vx.shape[0]:
                minlen = min(len(vy_reg_scaled), vx.shape[0])
                logging.warning(f"[ALIGN] vol-norm val y / x length mismatch (y:{len(vy_reg_scaled)} vs x:{vx.shape[0]}), trimming to {minlen}")
                vy_reg_scaled = vy_reg_scaled[:minlen]
                vx = vx[:minlen]
    # if not using vol-norm, use default z-scored targets
    if not use_vol_norm:
        ty_reg_scaled = ty_reg_scaled_default
        vy_reg_scaled = vy_reg_scaled_default
        volnorm_mean = None
        volnorm_std = None
        train_vol_pct = None
        test_vol_pct = None


    # 训练时使用增强数据
    x_aug, y_aug = ds.time_series_augmentation(tx, ty_reg_scaled, noise_level=0.01)

    if False:   #多分类时启用
        # 添加类别分布分析
        logging.info("=== Training Data Class Distribution ===")
        train_counts = np.bincount(ty[:,0], minlength=NUM_CLASSES)
        train_percent = train_counts / train_counts.sum()
        for i in range(NUM_CLASSES):
            logging.info(f"Class {i}: {train_counts[i]} samples ({train_percent[i]*100:.2f}%)")
        logging.info("=== Validation Data Class Distribution ===")
        val_counts = np.bincount(vy[:,0], minlength=NUM_CLASSES)
        val_percent = val_counts / val_counts.sum()
        for i in range(NUM_CLASSES):
            logging.info(f"Class {i}: {val_counts[i]} samples ({val_percent[i]*100:.2f}%)")
    
    if False:# 计算并打印特征相关性,多分类
        feature_data = ds.raw_train_x[-len(ds.train_y):]  # 裁剪到和train_y一致
        bin_labels = ds.train_y[:, 0]
        feature_names = ds.get_feature_names()
        selected = select_features_by_stat_corr(bin_labels, feature_data, feature_names, method='pearson', threshold=0.15)
        logging.info(f"皮尔逊筛选项: {selected}")
        selected = select_features_by_stat_corr(bin_labels, feature_data, feature_names, method='mi', threshold=0.03)
        logging.info(f"互信息筛选项: {selected}")
        bmgr = ds.bins1
        #bmgr.plot_bin_feature_correlation(bin_labels, feature_data, feature_names, save_path="bin_feature_corr.png")

    if False:# 计算并打印特征相关性,回归
        feature_data = ds.raw_train_x[-len(ds.train_y):]  # shape [n_samples, n_features]
        target = ds.train_y[:, 0]  # 回归目标（涨跌幅）
        feature_names = ds.get_feature_names()
        selected_intersection, _ = feature_importance_analysis(ds, feature_names, n_features=25)
        #selected = auto_select_features(feature_data, target, feature_names,
        #                            pearson_threshold=0.03, mi_threshold=0.01,
        #                            print_detail=True)
        #selected_rf, rf_scores = select_features_by_tree_importance(
        #    feature_data, target, feature_names,
        #    importance_threshold=0.01,
        #    print_detail=True
        #)
        #selected_intersection = set(selected['pearson_selected']) & set(selected['mi_selected']) & set(selected_rf)
        print("皮尔逊+互信息+树模型交集特征:", selected_intersection)
        exit()

    for batch_size in batch_size_list:
        for p in p_list:
            for epo in epo_list:
                #tm = LSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, p=p)
                tm = ResidualLSTMModel(x=tx, y=ty_reg_scaled, test_x=vx, test_y=vy_reg_scaled, p=p)
                #tm = TCNModel(x=tx, y=ty_reg_scaled, test_x=vx, test_y=vy_reg_scaled, nb_filters=nb_filters, kernel_size=kernel_size, dropout_rate=dropout_rate)
                #tm = TCNModel(x=x_aug, y=y_aug, test_x=vx, test_y=vy_reg_scaled, nb_filters=nb_filters, kernel_size=kernel_size, dropout_rate=dropout_rate)
                #tm = TransformerModel(tx, ty_reg_scaled, vx, vy_reg_scaled, d_model=d_model, num_layers=num_layers, ff_dim=ff_dim, dropout_rate=dropout_rate)
                print("################################ ### epo[%d] ### batch[%d] ### p[%d] ### ################################"%(epo, batch_size, p))
                train_ret_str = tm.train(epochs=epo, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                tm.save(os.path.join(BASE_DIR, MODEL_DIR, primary_stock_code + "_" + str(epo) + "_" + str(batch_size) + "_" + str(p) + ".h5"))
                logging.info(f"{train_ret_str}")

                if False:   #下面4行多分类时启用
                    best_val_t1 = tm.history.get_best_val()
                    last_loss, last_val_loss = tm.history.get_last_loss()
                    best_val_loss = tm.history.get_best_val_loss()
                    logging.info(f"INFO: last train/test/best_test loss - [{last_loss:.3f}]/[{last_val_loss:.3f}]/[{best_val_loss:.3f}],  best_test_accu T1 - [{best_val_t1:.1f}%]")
                print("*************************************************************************************************************************\n\n")
                # 训练模型后,下面5行为回归评估
                pred_scaled = tm.model.predict(vx).reshape(-1)

                if use_vol_norm:
                    # pred_scaled is z-score of (target_pct / vol_pct)
                    pred_volnorm = pred_scaled * volnorm_std + volnorm_mean  # target / vol_pct
                    # try to multiply back by per-window vol pct
                    if test_vol_pct is not None and test_vol_pct.size == pred_volnorm.size:
                        pred_pct = pred_volnorm * test_vol_pct
                    else:
                        avg_train_vol = np.mean(train_vol_pct) if (train_vol_pct is not None and train_vol_pct.size>0) else 1.0
                        logging.warning("test vol pct unavailable or length mismatch -> using avg train vol for scaling back")
                        pred_pct = pred_volnorm * avg_train_vol
                    # ground-truth percentage values:
                    y_true_pct = vy_reg[:len(pred_pct)] if vy_reg.size>0 else np.array([])
                else:
                    # original path - pred_scaled is z-scored of raw percentage target
                    pred_pct = pred_scaled * std_y + mean_y
                    y_true_pct = vy_reg[:len(pred_pct)] if vy_reg.size>0 else np.array([])

                # metrics
                if y_true_pct.size > 0:
                    mae = np.mean(np.abs(pred_pct - y_true_pct))
                    rmse = np.sqrt(np.mean((pred_pct - y_true_pct) ** 2))
                    logging.info(f"回归评估 (最终百分比空间): MAE={mae:.5f}, RMSE={rmse:.5f}")
                    try:
                        plot_regression_result(y_true_pct, pred_pct, title="test vs. real (pct)")
                        plot_error_distribution(y_true_pct, pred_pct, title="mae/rmse distribution (pct)")
                    except Exception:
                        pass
                else:
                    logging.warning("No ground truth percentages for evaluation.")
                print("*************************************************************************************************************************\n\n")

                # 训练模型后,输出给定日期的预测结果
                t_list = ['20250829', '20250901', '20250902', '20250903']
                for t0 in t_list:
                    print("Predict for T0[%s]"%t0)
                    # Ensure t0 matches the type of dataset dates
                    date_type = type(ds.full_raw_data[0, 0])
                    t0_converted = date_type(t0)
                    data, bp = ds.get_predictable_dataset_by_date(t0_converted)
                    pred_data = tm.model(data)
                    RegPredict(pred_data, bp, std_y, mean_y).print_predict_result()
                    print()

                # 训练模型后,输出训练曲线
                if False:
                    tm.plot()

                # 训练模型后,输出混淆矩阵
                if False:
                    y_pred = tm.model.predict(vx)
                    y_pred_label = np.argmax(y_pred, axis=1)
                    cm = confusion_matrix(vy[:, 0], y_pred_label)
                    plt.imshow(cm, cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.colorbar()
                    plt.show()

                del tm

