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
from utils.const_def import TOKEN, REL_CODE_LIST, NUM_CLASSES
from utils.const_def import BASE_DIR, MODEL_DIR
from utils.utils import setup_logging, StockType

if __name__ == "__main__":
    setup_logging()
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    epo_list = [100]
    #epo_list = [10,50,100]
    p_list = [4]
    #p_list = [2,16]
    batch_size_list = [32]
    #batch_size_list = [8,128,512,32]
    if_print_detail =False

    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    related_stock_list = REL_CODE_LIST
    ds = StockDataset(primary_stock_code, si, start_date='20070101',end_date='20250903', train_size=0.9)

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
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

                if True:
                    tm.plot()

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
