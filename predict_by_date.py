# coding=utf-8
import os
import sys
import argparse
import numpy as np
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.utils import get_model_file_name
from utils.tk import TOKEN
from utils.const_def import BASE_DIR, MODEL_DIR, NUM_CLASSES
from utils.utils import FeatureType, ModelType, PredictType, setup_logging
from model.residual_lstm import ResidualLSTMModel
from model.residual_tcn import ResidualTCNModel
from model.transformer import TransformerModel
from model.lstmmodel import LSTMModel
from predicproc.predict import Predict, RegPredict


def load_model_by_params(stock_code, model_type, predict_type, feature_type):
    model_fn = get_model_file_name(stock_code, model_type, predict_type, feature_type)
    if model_type == ModelType.RESIDUAL_LSTM:
        model = ResidualLSTMModel(fn=model_fn, predict_type=predict_type)
    elif model_type == ModelType.RESIDUAL_TCN:
        model = ResidualTCNModel(fn=model_fn, predict_type=predict_type)
    elif model_type == ModelType.TRANSFORMER:
        model = TransformerModel()
        model.load(model_fn)
    elif model_type == ModelType.MINI:
        model = LSTMModel(fn=model_fn, predict_type=predict_type)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Use trained model for prediction by date")
    parser.add_argument("--stock_code", default="600036.SH", help="Primary stock code")
    parser.add_argument("--feature_type", default="ALL", help="Feature type to use, e.g., ALL, T1L_25, T2H_25")
    parser.add_argument("--model_type", default="RESIDUAL_LSTM", choices=["RESIDUAL_LSTM", "RESIDUAL_TCN", "TRANSFORMER", "MINI"], help="Type of model")
    parser.add_argument("--predict_type", default="BINARY_T1_L10", help="PredictType, e.g., BINARY_T1_L10 or CLASSIFY")
    parser.add_argument("--dates", nargs="+", required=True, help="List of dates to predict, format: YYYYMMDD")
    parser.add_argument("--start_date", default="20190104", help="Start date for data")
    parser.add_argument("--end_date", default=None, help="End date for data")
    args = parser.parse_args()

    setup_logging()

    model_type = getattr(ModelType, args.model_type.upper(), ModelType.RESIDUAL_TCN)
    predict_type = getattr(PredictType, args.predict_type, PredictType.BINARY_T1_L10)
    feature_type = getattr(FeatureType, args.feature_type.upper(), FeatureType.T1L_35)
    model = load_model_by_params(args.stock_code, model_type, predict_type, feature_type)

    si = StockInfo(TOKEN)
    ds = StockDataset(
        ts_code=args.stock_code,
        idx_code_list=['000001.SH'],
        rel_code_list=[],
        si=si,
        start_date=args.start_date,
        end_date=args.end_date,
        train_size=0.99,
        feature_type=feature_type,
        if_update_scaler=False,
        predict_type=predict_type
    )

    for date_str in args.dates:
        print(f"\n==== T0:[{si.get_next_or_current_trade_date(date_str)}] / T1:[{si.get_next_trade_date(si.get_next_or_current_trade_date(date_str))}] / T2:[{si.get_next_trade_date(si.get_next_trade_date(si.get_next_or_current_trade_date(date_str)))}] ====")
        try:
            x_input, base_price = ds.get_predictable_dataset_by_date(date_str)
        except Exception as e:
            print(f"日期 {date_str} 数据不可用: {e}")
            continue
        pred = model.model.predict(x_input, verbose=0)

        # 分类/二分类/回归均用Predict类输出
        if predict_type.is_classify():
            Predict(pred, base_price, predict_type, ds.bins1, ds.bins2).print_predict_result("预测")
        elif predict_type.is_binary():
            Predict(pred, base_price, predict_type).print_predict_result("预测")
        elif predict_type.is_regress():
            RegPredict(pred, base_price).print_predict_result("预测")
        else:
            print(f"预测值: {pred}")

        # 是否输出真实结果对比  
        idx = np.where(ds.date_list == int(date_str))[0]
        is_historical = len(idx) > 0 and idx[0] < len(ds.raw_y)
        if is_historical:
            real_raw_y = ds.raw_y[idx[0]]
            if predict_type.is_binary_t1_low():
                real_y = (real_raw_y[0]*100 <= predict_type.val).astype(int).reshape(-1, 1)
            elif predict_type.is_binary_t1_high():
                real_y = (real_raw_y[1]*100 >= predict_type.val).astype(int).reshape(-1, 1)
            elif predict_type.is_binary_t2_low():
                real_y = (real_raw_y[2]*100 <= predict_type.val).astype(int).reshape(-1, 1)
            elif predict_type.is_binary_t2_high():
                real_y = (real_raw_y[3]*100 >= predict_type.val).astype(int).reshape(-1, 1)

            # 用Predict类处理真实标签
            Predict.from_real_label(real_y, base_price, predict_type, ds.bins1, ds.bins2).print_predict_result("真实")
        else:
            print(f"无真实标签，仅输出预测结果。")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()