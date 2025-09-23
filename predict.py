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
from predicproc.predict import Predict, RegPredict
from model.residual_lstm import ResidualLSTMModel
from bins import BinManager
from utils.tk import TOKEN
from utils.const_def import BASE, T1L_SCALE, T2H_SCALE, REL_CODE_LIST, NUM_CLASSES

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    si = StockInfo(TOKEN)
    primary_stock = '600036.SH'
    related_stock_list = REL_CODE_LIST
    ps = Stock(primary_stock, si, start_date='20070101',  if_force_download=False)
    sd = StockDataset(ps)

    bins1 = BinManager(BASE+"cfg\\600036.SH_pri_y1_bins.json")
    bins2 = BinManager(BASE+"cfg\\600036.SH_pri_y2_bins.json")

    tm = ResidualLSTMModel(fn = BASE + "\\model\\temp_best.h5")
    t_list = ['20250829', '20250901', '20250902']
    
    if False:
        #多分类预测
        for t0 in t_list:
            print("Predict for T0[%s]"%t0)
            data, bp = sd.get_predictable_dataset_by_date(t0)
            pred_data = tm.model(data)
            Predict(pred_data, bp, bins1.bins, bins2.bins).print_predict_result()
            print()

    else:
        #回归预测
        for t0 in t_list:
            print("Predict for T0[%s]"%t0)
            data, bp = sd.get_predictable_dataset_by_date(t0)
            pred_data = tm.model.predict(data)  # shape: (1, 1)
            RegPredict(pred_data, bp).print_predict_result()
            print()