# coding=utf-8
import os, sys
import argparse, warnings
import numpy as np
from datetime import datetime
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.utils import load_model_by_params
from utils.tk import TOKEN
from utils.const_def import BASE_DIR, MODEL_DIR, NUM_CLASSES, IDX_CODE_LIST
from utils.utils import FeatureType, ModelType, PredictType, setup_logging
from predicproc.predict import Predict, RegPredict

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Use trained model for prediction by date")
    parser.add_argument("--stock_code", default="600036.SH", help="Primary stock code")
    parser.add_argument("--feature_type", default="T1L10_F55", help="Feature type to use, e.g., ALL, T1L_25, T2H_25")
    parser.add_argument("--model_type", default="RESIDUAL_LSTM", choices=["RESIDUAL_LSTM", "RESIDUAL_TCN", "TRANSFORMER", "MINI"], help="Type of model")
    parser.add_argument("--predict_type", default="BINARY_T1_L10", help="PredictType, e.g., BINARY_T1_L10 or CLASSIFY")
    parser.add_argument("--dates", nargs="+", required=False, help="List of dates to predict, format: YYYYMMDD")
    parser.add_argument("--from_date", default=None, help="If set, predict all trade dates from this date to today")
    parser.add_argument("--start_date", default="20190104", help="Start date for data")
    parser.add_argument("--end_date", default=None, help="End date for data")
    args = parser.parse_args()

    setup_logging()

    model_type = getattr(ModelType, args.model_type.upper(), ModelType.RESIDUAL_LSTM)
    predict_type = getattr(PredictType, args.predict_type, PredictType.BINARY_T1_L10)
    feature_type = getattr(FeatureType, args.feature_type.upper(), FeatureType.T1L10_F55)
    model = load_model_by_params(args.stock_code, model_type, predict_type, feature_type)
    print(f"model_type: {model_type}, predict_type: {predict_type}, feature_type: {feature_type}")

    if args.from_date is None and args.dates is None:
        print("错误: 必须指定 --dates 或 --from_date")
        sys.exit(1)
    elif args.from_date is not None:
        today = int(datetime.now().strftime('%Y%m%d'))
        si = StockInfo(TOKEN)
        trade_dates_df = si.get_trade_open_dates(int(args.from_date), today)
        args.dates = trade_dates_df['trade_date'].astype(int).tolist()
        print(f"预测从 {args.from_date} 到今天的所有交易日，共 {len(args.dates)} 天: {args.dates}")
    else:
        print(f"预测指定的日期: {args.dates}")

    si = StockInfo(TOKEN)
    ds = StockDataset(idx_code_list=IDX_CODE_LIST, rel_code_list=[], si=si, train_size=1, if_update_scaler=False,
        ts_code=args.stock_code,        
        start_date=args.start_date,
        end_date=args.end_date,
        feature_type=feature_type,
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
        #else:
        #    print(f"无真实标签，仅输出预测结果。")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()