# coding=utf-8
import os, sys, time, argparse, datetime, logging
import numpy as np
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from datasets.cat import RateCat
from datasets.stock import Stock
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from predicproc.predict import Predict, RegPredict
from model.lstmmodel import LSTMModel
from sklearn.metrics import confusion_matrix
from utils.tk import TOKEN
from utils.const_def import REL_CODE_LIST, NUM_CLASSES, T1L_SCALE, T2H_SCALE
from utils.const_def import BASE_DIR, MODEL_DIR
from utils.utils import setup_logging, select_features_by_tree_importance, auto_select_features, select_features_by_stat_corr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    setup_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # 优化后的训练参数 - 使用更多的epoch和更好的batch size
    epo_list = [100]  # 增加epochs，早停会自动停止
    p_list = [1]
    batch_size_list = [64]  # 增加batch size以提高训练稳定性
    learning_rate = 0.00005  # 使用更高的初始学习率
    patience = 100  # 提高早停的耐心值，允许更多epoch的波动
    if_print_detail = False

    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = ['000001.SH']#, '399001.SZ', '399006.SZ']  #上证指数,深证成指,创业板指
    related_stock_list = REL_CODE_LIST
    # 改善数据集配置 - 使用更好的train/validation分割比例
    ds = StockDataset(primary_stock_code, index_code_list, si, start_date='20070101',end_date='20250903', train_size=0.8)  # 85%/15%分割提供更多验证数据

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    ### 只用T1 low的涨跌幅为回归目标 ###
    ty_reg = ty[:, 0].astype(float)
    vy_reg = vy[:, 0].astype(float)
    
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
        selected = auto_select_features(feature_data, target, feature_names,
                                    pearson_threshold=0.05, mi_threshold=0.02,
                                    print_detail=True)
        selected_rf, rf_scores = select_features_by_tree_importance(
            feature_data, target, feature_names,
            importance_threshold=0.015,
            print_detail=True
        )
        selected_intersection = set(selected['pearson_selected']) & set(selected['mi_selected']) & set(selected_rf)
        print("皮尔逊+互信息+树模型交集特征:", selected_intersection)
        exit()

    for batch_size in batch_size_list:
        for p in p_list:
            for epo in epo_list:
                tm = LSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, p=p)
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
                y_pred = tm.model.predict(vx).reshape(-1)
                mae = np.mean(np.abs(y_pred - vy_reg))
                rmse = np.sqrt(np.mean((y_pred - vy_reg) ** 2))
                logging.info(f"回归评估: MAE={mae:.5f}, RMSE={rmse:.5f}")
                print("*************************************************************************************************************************\n\n")

                # 训练模型后,输出给定日期的预测结果
                t_list = ['20250829', '20250901', '20250902', '20250903']
                for t0 in t_list:
                    print("Predict for T0[%s]"%t0)
                    data, bp = ds.get_predictable_dataset_by_date(t0)
                    pred_data = tm.model(data)
                    RegPredict(pred_data, bp).print_predict_result()
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

