# coding=utf-8
from datetime import datetime, timedelta
from math import ceil
from enum import Enum, auto
import logging.config


def setup_logging():
    logging.basicConfig(
        #level=logging.DEBUG,
        level=logging.INFO,
        format='%(levelname)s %(message)s',
        #format='%(asctime)s %(levelname)s %(message)s',
        #format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        #format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d %(funcName)s] %(message)s',
        datefmt='%H:%M:%S'
    )

def get_predict_score(pred, real):
    score = 0
    if pred == real:    #完全预测正确
        score = 1       #则可得满分
    #elif abs(real-pred) == 1: #预测范围空间差距1档
    #    score = 0.5           #预测范围空间差距1档，得0.5分

    #elif (real<=4 and pred<=4) or (real>=5 and pred>=5):    #预测涨跌正确
    #    score = score + 0.5
    #    if abs(real-pred) == 1:     #预测范围空间差距1档
    #        score = score + 0.25
    #    elif abs(real-pred) == 2:   #预测范围空间差距2档
    #        score = score + 0.1
    #    else:
    #        pass
    else:
        pass

    return score

def get_minor_one_day(input_date):
    ret_date =  get_datetime_date(input_date) - timedelta(days=1)
    ret_date_in_string = get_string_date(ret_date)
    #print("INFO: get_minor_one_day() - in date is :<%s>, ret date is :<%s>"%(input_date, ret_date_in_string))
    return ret_date_in_string

def get_datetime_date(input_date):
    return datetime.strptime(input_date,"%Y%m%d")

def day_plus_minor(input_date, dd):
    id = get_datetime_date(input_date)
    return (id+timedelta(days=dd)).strftime("%Y%m%d")

def get_string_date(input_datetime):
    return input_datetime.strftime("%Y%m%d")

def get_mind_value(value, base_value):
    if value <= base_value:
        if ceil(value) - value < 0.02:
            return ceil(value) + 0.03
    else:
        if value - int(value) < 0.02:
            return int(value) - 0.03
    return value

    
class SuperList(list):
    def append(self, item):
        super().append(item)
        return self


class StockType(Enum):
    PRIMARY = auto()
    RELATED = auto()
    INDEX = auto()