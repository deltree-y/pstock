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
from predicproc.predict import Predict
from model.lstmmodel import LSTMModel
from sklearn.metrics import confusion_matrix
from utils.tk import TOKEN
from utils.const_def import REL_CODE_LIST, NUM_CLASSES
from utils.const_def import BASE_DIR, MODEL_DIR
from utils.utils import setup_logging, StockType
import matplotlib.pyplot as plt

if __name__ == "__main__":
    setup_logging()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # 优化后的训练参数 - 使用更多的epoch和更好的batch size
    epo_list = [300]  # 增加epochs，早停会自动停止
    p_list = [8]
    batch_size_list = [64]  # 增加batch size以提高训练稳定性
    if_print_detail = False

    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = ['000001.SH']#, '399001.SZ', '399006.SZ']  #上证指数,深证成指,创业板指
    related_stock_list = REL_CODE_LIST
    # 改善数据集配置 - 使用更好的train/validation分割比例
    ds = StockDataset(primary_stock_code, index_code_list, si, start_date='20070101',end_date='20250903', train_size=0.8)  # 85%/15%分割提供更多验证数据

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    
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
    
    for batch_size in batch_size_list:
        for p in p_list:
            for epo in epo_list:
                tm = LSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, p=p)
                print("################################ ### epo[%d] ### batch[%d] ### p[%d] ### ################################"%(epo, batch_size, p))
                train_ret_str = tm.train(epochs=epo, batch_size=batch_size)
                tm.save(os.path.join(BASE_DIR, MODEL_DIR, primary_stock_code + "_" + str(epo) + "_" + str(batch_size) + "_" + str(p) + ".h5"))
                best_val_t1 = tm.history.get_best_val()
                last_loss, last_val_loss = tm.history.get_last_loss()
                best_val_loss = tm.history.get_best_val_loss()

                logging.info(f"{train_ret_str}")
                logging.info(f"INFO: last train/test/best_test loss - [{last_loss:.3f}]/[{last_val_loss:.3f}]/[{best_val_loss:.3f}],  best_test_accu T1 - [{best_val_t1:.1f}%]")
                print("*************************************************************************************************************************\n\n")
                t_list = ['20250829', '20250901']
                for t0 in t_list:
                    print("Predict for T0[%s]"%t0)
                    data, bp = ds.get_predictable_dataset_by_date(t0)
                    pred_data = tm.model(data)
                    Predict(pred_data, bp, ds.bins1.prop_bins, ds.bins2.prop_bins).print_predict_result()
                    print()

                if False:
                    tm.plot()

                # 训练模型后
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



    if False:
        #sds.save_raw(BASE + "\\temp\\raw_x.csv")
        counts = np.bincount(ty[:,0], minlength=NUM_CLASSES)
        percent = counts / counts.sum()
        for i in range(NUM_CLASSES):
            print(f"Label {i}: {counts[i]} ({percent[i]*100:.2f}%)")
        print()
        counts = np.bincount(ty[:,1], minlength=NUM_CLASSES)
        percent = counts / counts.sum()
        for i in range(NUM_CLASSES):
            print(f"Label {i}: {counts[i]} ({percent[i]*100:.2f}%)")
        exit()
