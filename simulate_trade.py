# coding=utf-8
import os
import sys
import argparse
import warnings
import numpy as np
from datetime import datetime, timedelta

# --------- 项目内路径与导入 ----------
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.utils import load_model_by_params
from predicproc.predict import Predict, RegPredict
from datasets.money import Funds
from utils.tk import TOKEN
from utils.const_def import IDX_CODE_LIST, CONTINUOUS_DAYS
from utils.utils import FeatureType, ModelType, PredictType, setup_logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _add_days(date_str, days):
    d = datetime.strptime(str(date_str), "%Y%m%d")
    d2 = d + timedelta(days=days)
    return d2.strftime("%Y%m%d")


def build_ds_and_model(stock_code, feature_type, model_type, predict_type,
                       backtest_start, backtest_end, train_size=1.0):
    """
    注意：数据集的 start_date 会比回测起始日更早，以满足窗口长度 CONTINUOUS_DAYS 的需要。
    但真正做交易的日期只用 [backtest_start, backtest_end]。
    """
    si = StockInfo(TOKEN)

    # 为构造窗口，实际数据起始日提前 CONTINUOUS_DAYS+若干天
    ds_start_date = _add_days(backtest_start, -CONTINUOUS_DAYS - 5)
    ds_end_date = backtest_end

    ds = StockDataset(
        ts_code=stock_code,
        idx_code_list=IDX_CODE_LIST,
        rel_code_list=[],
        si=si,
        start_date=ds_start_date,
        end_date=ds_end_date,
        train_size=train_size,
        feature_type=feature_type,
        predict_type=predict_type,
        if_update_scaler=False  # 回测时直接用已训练好的 scaler
    )
    model = load_model_by_params(stock_code, model_type, predict_type, feature_type)
    return si, ds, model


def generate_signal(pred_obj: Predict, threshold=0.5):
    """
    根据预测结果产生交易信号：
      - 返回值: 'BUY' / 'SELL' / 'HOLD'
    """
    if pred_obj.is_binary:
        pt = pred_obj.predict_type
        # T1L跌,且T2H涨 -> 买入
        if pt.is_binary_t1_low():
            return 'SELL' if pred_obj.pred_label == 1 else 'BUY'
        # 涨破类（二分类 T1_H/T2_H）：预测会涨 -> BUY，否则 SELL
        elif pt.is_binary_t1_high() or pt.is_binary_t2_high():
            return 'BUY' if pred_obj.pred_label == 1 else 'SELL'
        else:
            return 'HOLD'

    elif pred_obj.is_classify:
        # 多分类：高类别视为看多，低类别看空
        label = int(pred_obj.y1r.get_label())
        num_classes = pred_obj.bins1.n_bins + 1
        mid = num_classes // 2
        return 'BUY' if label >= mid else 'SELL'
    else:
        # 回归：涨幅>0 买入，否则卖出
        return 'BUY' if pred_obj.pred_value > 0 else 'SELL'


def simulate_trading(stock_code, feature_type, model_type, predict_type,
                     start_date, end_date, init_capital=500000,
                     use_buy_max=True):
    """
    回测区间严格为 [start_date, end_date]，只遍历这一段的交易日。
    但 StockDataset 内部会自动往前扩展数据起点，用于构造窗口。
    """
    setup_logging()
    warnings.filterwarnings("ignore", category=UserWarning)

    backtest_start, backtest_end = start_date, end_date

    # ---- 构建数据集&模型（数据起点提前，但只在回测区间内交易） ----
    si, ds, model = build_ds_and_model(
        stock_code=stock_code,
        feature_type=feature_type,
        model_type=model_type,
        predict_type=predict_type,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        train_size=1.0
    )

    # ---- 获取“只在回测区间内”的交易日序列 ----
    trade_dates_df = si.get_trade_open_dates(backtest_start, backtest_end)
    date_list = trade_dates_df['trade_date'].astype(str).tolist()  # 已按从旧到新排序

    # 资金账户
    f = Funds(init_amount=init_capital)

    # 记录曲线
    equity_curve = []
    equity_dates = []

    # 统计
    trade_log = []

    # 从旧到新遍历【严格只在用户指定区间内】
    for d in reversed(date_list):
        # 尝试获得当天可预测的数据窗口和基准价
        try:
            x_input, base_price = ds.get_predictable_dataset_by_date(d)
        except Exception:
            # 该日无法构造完整窗口（即使前面多取了历史，也可能前几天太靠近数据起点），跳过
            continue

        # 使用模型预测
        pred_raw = model.model.predict(x_input, verbose=0)

        # 构造 Predict 对象（统一处理多分类/二分类/回归）
        if predict_type.is_classify():
            pred_obj = Predict(pred_raw, base_price, predict_type, ds.bins1, ds.bins2)
        elif predict_type.is_binary():
            pred_obj = Predict(pred_raw, base_price, predict_type)
        elif predict_type.is_regress():
            pred_obj = RegPredict(pred_raw, base_price)
        else:
            continue

        # 产生交易信号
        signal = generate_signal(pred_obj)

        # 执行交易决策（简单示例，仅允许全仓买入/全仓卖出）
        if signal == 'BUY':
            if use_buy_max:
                qty = f.buy_max(base_price, date=d, is_print=False)
            else:
                qty = f.buy_stock(base_price, 100, date=d)  # 只买一手示例
            if qty > 0:
                trade_log.append(
                    (d, 'BUY', base_price, qty, f.get_total_amount(base_price))
                )
        elif signal == 'SELL':
            if f.get_stock_quantity() > 0:
                qty = f.sell_all(base_price, date=d, is_print=False)
                if qty > 0:
                    trade_log.append(
                        (d, 'SELL', base_price, qty, f.get_total_amount(base_price))
                    )

        # 每日权益记录：用当前收盘价估算
        total_equity = f.get_total_amount(base_price)
        equity_curve.append(total_equity)
        equity_dates.append(d)

    # 回测结束，若还有持仓，可选择按最后一天价格平仓
    if f.get_stock_quantity() > 0 and len(equity_dates) > 0:
        last_date = equity_dates[-1]
        try:
            _, last_price = ds.get_predictable_dataset_by_date(last_date)
        except Exception:
            last_price = base_price
        f.sell_all(last_price, date=last_date, is_print=False)
        total_equity = f.get_total_amount(last_price)
        equity_curve[-1] = total_equity  # 更新最后一天权益

    # 结果统计
    final_equity = equity_curve[-1] if equity_curve else init_capital
    total_return = (final_equity - init_capital) / init_capital * 100

    print("=" * 80)
    print(f"回测股票: {stock_code}")
    print(f"模型: {model_type}, 特征: {feature_type}, 预测类型: {predict_type}")
    print(f"数据起点(内部实际使用): {ds.stock.start_date}")
    print(f"回测区间(实际遍历): [{backtest_start}] - [{backtest_end}]")
    print(f"初始资金: {init_capital:.2f}")
    print(f"结束资金: {final_equity:.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"交易次数: {len(trade_log)}")
    print("=" * 80)

    # 简单打印部分交易记录
    for rec in trade_log:#[:10]:
        d, side, price, qty, equity = rec
        print(f"{d} {side:4s} @ {price:.2f}, qty={qty}, equity={equity:.2f}")
    if len(trade_log) > 10:
        print(f"... 共 {len(trade_log)} 笔交易")#，仅展示前 10 笔")

    # 返回曲线和日志，方便后续画图或做更深入分析
    return {
        "dates": equity_dates,
        "equity": equity_curve,
        "trades": trade_log,
        "final_equity": final_equity,
        "total_return": total_return
    }


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的模型，在指定日期区间内模拟交易并评估收益")
    parser.add_argument("--stock_code", default="600036.SH", help="主股票代码")
    parser.add_argument("--feature_type", default="CLASSIFY_F50", help="FeatureType 枚举名，如 T1L10_F55 / CLASSIFY_F50 等")
    parser.add_argument("--model_type", default="TRANSFORMER",
                        choices=["RESIDUAL_LSTM", "RESIDUAL_TCN", "TRANSFORMER", "MINI", "CONV1D"],
                        help="ModelType 枚举名")
    parser.add_argument("--predict_type", default="CLASSIFY_F50", help="PredictType 枚举名，比如 BINARY_T1_L10 / CLASSIFY")
    parser.add_argument("--start_date", default="20250101", help="回测开始日期，YYYYMMDD")
    parser.add_argument("--end_date", default=None, help="回测结束日期，YYYYMMDD，默认到今天")
    parser.add_argument("--init_capital", type=float, default=500000, help="初始资金")
    parser.add_argument("--no_buy_max", action="store_true", help="不满仓买入，每次只买一手")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 转成枚举
    model_type = getattr(ModelType, args.model_type.upper(), ModelType.TRANSFORMER)
    predict_type = getattr(PredictType, args.predict_type, PredictType.CLASSIFY)
    feature_type = getattr(FeatureType, args.feature_type.upper(), FeatureType.CLASSIFY_F50)

    end_date = args.end_date or datetime.now().strftime('%Y%m%d')

    simulate_trading(
        stock_code=args.stock_code,
        feature_type=feature_type,
        model_type=model_type,
        predict_type=predict_type,
        start_date=args.start_date,
        end_date=end_date,
        init_capital=args.init_capital,
        use_buy_max=True#not args.no_buy_max
    )