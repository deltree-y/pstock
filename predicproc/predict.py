#-*- coding:UTF-8 -*-
import sys, os, logging
import numpy as np
from pathlib import Path
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
from cat import RateCat

class Predict():
    def __init__(self, predicted_data, base_price, bins1, bins2):
        self.predicted_data = predicted_data
        self.bp = base_price
        self.bins1, self.bins2 = bins1, bins2
        self.y1r = RateCat(one_hot=self.predicted_data[0,:], scale=self.bins1, tag='T1L')
        #self.y2r = RateCat(one_hot=self.predicted_data[1,:], scale=self.bins1, tag='T2H')

    def print_predict_result(self):
        logging.info(f"Predict t0p[{self.bp}] y1l label[{self.y1r.get_label()}] pct min/avg/max is <{self.y1r.get_rate_from_label('min')*100:.2f}%%/{self.y1r.get_rate_from_label('avg')*100:.2f}%%/{self.y1r.get_rate_from_label('max')*100:.2f}%%> price is <{self.y1r.get_rate_from_label('min')*self.bp+self.bp:.2f}/{self.y1r.get_rate_from_label('avg')*self.bp+self.bp:.2f}/{self.y1r.get_rate_from_label('max')*self.bp+self.bp:.2f}>")
        #logging.info("INFO: Predict y2h min/avg/max is <%.3f/%.3f/%.3f>" % (self.y2r.get_rate_from_label('min'), self.y2r.get_rate_from_label('avg'), self.y2r.get_rate_from_label('max')))

