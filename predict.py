# coding=utf-8
import os, sys, time, argparse, datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from datasets.cat import RateCat
from datasets.stock import Stock, Stocks
from datasets.stockinfo import StockInfo
from dataset import StockDataset, StockDatasets
from predicproc.predict import Predict
from model.lstmmodel import LSTMModel
from bins import BinManager
from utils.const_def import BASE, T1L_SCALE, T2H_SCALE, TOKEN, REL_CODE_LIST, NUM_CLASSES

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    epo_list = [2]
    #epo_list = [10,50,100]
    p_list = [2]
    #p_list = [1,2,4,8,16,32,64]
    #batch_size_list = [8,16,64,256]
    batch_size_list = [16]
    if_print_detail =False

    si = StockInfo(TOKEN)
    primary_stock = '600036.SH'
    related_stock_list = REL_CODE_LIST
    ps = Stock(primary_stock, si, start_date='20070101',  if_force_download=False)
    #rs = Stocks(related_stock_list, si, start_date='20150101', end_date='20250828', if_force_download=False)
    sd = StockDataset(ps)
    #sds = StockDatasets(ps, rs, if_update_scaler=False)

    bins1 = BinManager(BASE+"cfg\\600036.SH_pri_y1_bins.json")
    bins2 = BinManager(BASE+"cfg\\600036.SH_pri_y2_bins.json")

    tm = LSTMModel(fn = BASE + "\\model\\temp_best.h5")
    t_list = ['20250829', '20250901', '20250902']
    for t0 in t_list:
        print("Predict for T0[%s]"%t0)
        data, bp = sd.get_predictable_dataset_by_date(t0)
        pred_data = tm.model(data)
        Predict(pred_data, bp, bins1.prop_bins, bins2.prop_bins).print_predict_result()
        print()
