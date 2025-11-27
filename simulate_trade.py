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
from datasets.money import Funds
from utils.tk import TOKEN
from utils.const_def import IDX_CODE_LIST, CONTINUOUS_DAYS
from utils.utils import FeatureType, ModelType, PredictType, setup_logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _add_days(date_str, days):
    d = datetime.strptime(str(date_str), "%Y%m%d")
    d2 = d + timedelta(days=days)
    return d2.strftime("%Y%m%d")


def build_ds_and_model(
    stock_code: str,
    feature_type: FeatureType,
    model_type: ModelType,
    predict_type: PredictType,
    backtest_start: str,
    backtest_end: str,
    train_size: float = 1.0,
):
    """
    为某一套 (feature_type, predict_type) 构建独立的数据集和模型。
    注意：start_date 会比回测起始日更早，以满足 CONTINUOUS_DAYS。
    """
    si = StockInfo(TOKEN)

    ds_start_date = _add_days(backtest_start, -3*CONTINUOUS_DAYS)
    ds_end_date = backtest_end
    print(f"[构建数据集] 股票: {stock_code}, 特征: {feature_type}, 预测: {predict_type}, 数据区间: [{ds_start_date}] - [{ds_end_date}]")
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
        if_update_scaler=False,  # 回测时直接用已训练好的 scaler
    )
    model = load_model_by_params(stock_code, model_type, predict_type, feature_type)
    return si, ds, model


def _predict_binary_label(model, x_input):
    """对二分类模型做预测, 返回 (prob, label)"""
    raw = model.model.predict(x_input, verbose=0)
    prob = float(raw[0, 0])
    label = int(prob > 0.5)
    return prob, label


def _get_raw_index_by_date(ds: StockDataset, date_str: str) -> int:
    """
    在 ds.raw_data 中找到指定日期行的索引。
    raw_data 第0列是日期，类型通常是 int64/str，保持一致即可。
    """
    date_val = type(ds.raw_data[0, 0])(date_str)
    idx_arr = np.where(ds.raw_data[:, 0] == date_val)[0]
    if idx_arr.size == 0:
        raise ValueError(f"date {date_str} not found in raw_data")
    return int(idx_arr[0])


def _get_real_t1_prices(ds: StockDataset, date_str: str):
    """
    给定 T0 日期，返回真实 T1L / T1H 价格（来自 raw_data 的下一行 low/high）：

      - 在 raw_data 里找到 T0 所在行 i0；
      - T1 即 i0+1 行；
      - 真实 T1L = raw_data[i0+1, col_low+1]；
      - 真实 T1H = raw_data[i0+1, col_high+1]。

    若 i0+1 超出范围，说明没有 T1，返回 (None, None)。
    """
    i0 = _get_raw_index_by_date(ds, date_str)
    i1 = i0 + 1
    if i1 >= ds.raw_data.shape[0]:
        return None, None

    col_low = ds.p_trade.col_low + 1   # +1 是因为 raw_data 第0列是日期
    col_high = ds.p_trade.col_high + 1

    t1l = float(ds.raw_data[i1, col_low])
    t1h = float(ds.raw_data[i1, col_high])
    return t1l, t1h


def simulate_trading(
    stock_code: str,
    model_type: ModelType,
    # 买入信号用的二分类 + 特征
    t1l_buy_type: PredictType,      # BINARY_T1_Lxx
    t1l_buy_feature: FeatureType,   # 如 FeatureType.T1L10_F55
    t2h_sell_type: PredictType,      # BINARY_T2_Hxx
    t2h_sell_feature: FeatureType,   # 如 FeatureType.T2H10_F55
    # 卖出信号/挂单价用的二分类 + 特征
    t1h_sell_type: PredictType,     # BINARY_T1_Hzz （用来判断是否卖出 + 卖出挂单价）
    t1h_sell_feature: FeatureType,  # 对应 T1H 的特征
    start_date: str,
    end_date: str,
    init_capital: float = 500000,
    use_buy_max: bool = True,
):
    """
    统一整合后的策略（按你最终描述）：

    1. 买入决策（挂 T1L 买单）——“T1 低值跌，T2 高值涨”：
       - 使用 (t1l_buy_type, t1l_buy_feature) + (t2h_buy_type, t2h_buy_feature)：
         若 T1L=1 且 T2H=1 -> 产生买入意向；
       - 买入挂单价：buy_price = base_price * (1 + t1l_buy_type.val/100)
         （例如 BINARY_T1_L10, val=-1.0 => buy_price = base_price*0.99）；
       - 真实 T1L 价格 real_t1l_price 直接用“主 T1L 数据集 ds_t1l_buy” raw_data 中 T1 日 low；
       - 若 real_t1l_price <= buy_price，则认为买入挂单成交，以 buy_price 成交；
       - 默认 buy_max。

    2. 卖出决策（挂 T1H 卖单）——“手中有股即以 T1 高值卖出（模拟挂单可能成交/失败）”：
       - 若当前有持仓：
         a) 使用 (t1h_sell_type, t1h_sell_feature) 做二分类：
            若 T1H=1 才产生卖出意向；
         b) 卖出挂单价：sell_price = base_price * (1 + t1h_sell_type.val/100)
            （例如 BINARY_T1_H10, val=+1.0 => sell_price = base_price*1.01）；
         c) 真实 T1H 价格 real_t1h_price 直接用 ds_t1l_buy raw_data 中 T1 日 high；
         d) 若 real_t1h_price >= sell_price，则认为卖出挂单成交，以 sell_price 成交；
            否则卖出失败，继续持有。

    3. Feature 使用：
       - 四个二分类模型各用各自的 FeatureType：
         * T1L 买入模型：t1l_buy_feature
         * T2H 买入模型：t2h_buy_feature
         * T1H 卖出信号/挂单价模型：t1h_sell_feature
       - 真实价判断统一使用 “主 T1L 买入数据集 ds_t1l_buy” 的 raw_data 时间轴（保证 T0/T1 对齐）。

    4. 默认 buy_max 满仓买入。
    """
    setup_logging()
    warnings.filterwarnings("ignore", category=UserWarning)

    backtest_start, backtest_end = start_date, end_date

    # ===== 构建四套 (feature, predict_type) 的数据集 & 模型 =====
    # 选 T1L 买入数据集 ds_t1l_buy 作为“主时间轴”，用它的 raw_data 取 T1L/T1H 真实价格
    si, ds_t1l_buy, m_t1l_buy = build_ds_and_model(
        stock_code=stock_code,
        feature_type=t1l_buy_feature,
        model_type=model_type,
        predict_type=t1l_buy_type,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        train_size=1.0,
    )
    _, ds_t2h_sell, m_t2h_sell = build_ds_and_model(
        stock_code=stock_code,
        feature_type=t2h_sell_feature,
        model_type=model_type,
        predict_type=t2h_sell_type,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        train_size=1.0,
    )
    _, ds_t1h_sell, m_t1h_sell = build_ds_and_model(
        stock_code=stock_code,
        feature_type=t1h_sell_feature,
        model_type=model_type,
        predict_type=t1h_sell_type,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        train_size=1.0,
    )

    # ===== 回测日期（统一用 StockInfo 的交易日列表） =====
    trade_dates_df = si.get_trade_open_dates(backtest_start, backtest_end)
    date_list = trade_dates_df["trade_date"].astype(str).tolist()  # 从旧到新

    f = Funds(init_amount=init_capital)

    equity_curve = []
    equity_dates = []
    trade_log = []
    have_position = False

    for d in reversed(date_list):
        qty = 0
        print("回测日期:", d, end='')
        # ---- 针对 T0，在四个数据集上分别构造窗口 ----
        # 若任一关键数据集当天窗口不可用，则直接跳过该日
        try:
            x_t1l_buy, bp_t1l_buy = ds_t1l_buy.get_predictable_dataset_by_date(d)
            x_t2h_sell, _          = ds_t2h_sell.get_predictable_dataset_by_date(d)
            x_t1h_sell, _         = ds_t1h_sell.get_predictable_dataset_by_date(d)
        except Exception:
            continue

        # 记账 & 真实价用 base_price：统一采用主 T1L 买入数据集的基准价（T0 收盘价）
        base_price = float(bp_t1l_buy)

        # ====================== 卖出逻辑 ======================
        if have_position and f.get_stock_quantity() > 0:
            # 1) 用 T1H 卖出模型判断是否有卖出意向
            prob_t1h, label_t1h = _predict_binary_label(m_t1h_sell, x_t1h_sell)

            if label_t1h == 1:
                # 2) 卖出挂单价（由 BINARY_T1_Hzz 的阈值决定）
                sell_rate = t1h_sell_type.val / 100.0  # 正数，如 +1.0 -> 0.01
                sell_price = base_price * (1 + sell_rate)

                # 3) 从“主 T1L 数据集”的 raw_data 中获取真实 T1H 价格
                _, real_t1h_price = _get_real_t1_prices(ds_t1l_buy, d)

                if real_t1h_price is not None and real_t1h_price >= sell_price:
                    qty = f.sell_all(sell_price, date=d, is_print=False)
                    if qty > 0:
                        trade_log.append(
                            (
                                d,
                                "SELL",
                                sell_price,
                                qty,
                                f.get_total_amount(sell_price),
                                base_price,
                                real_t1h_price,
                                prob_t1h,
                            )
                        )
                        have_position = False

        # ====================== 买入逻辑 ======================
        # 1) T1L/T2H 二分类给出买入意向（“T1低值跌，T2高值涨”）
        prob_t1l_buy, label_t1l_buy = _predict_binary_label(m_t1l_buy, x_t1l_buy)
        prob_t2h_sell, label_t2h_sell = _predict_binary_label(m_t2h_sell, x_t2h_sell)

        has_buy_intent = (label_t1l_buy == 1) and (label_t2h_sell == 1)

        if has_buy_intent and f.get_stock_quantity() == 0:
            # 2) 买入挂单价（由 BINARY_T1_Lxx 的阈值决定）
            buy_rate = t1l_buy_type.val / 100.0  # 通常为负，例如 -1.0 -> -0.01
            buy_price = base_price * (1 + buy_rate)

            # 3) 真实 T1L 价格来自“主 T1L 数据集”的 raw_data T1 日 low
            real_t1l_price, _ = _get_real_t1_prices(ds_t1l_buy, d)

            if real_t1l_price is not None and real_t1l_price <= buy_price:
                # 挂单成交
                if use_buy_max:
                    qty = f.buy_max(buy_price, date=d, is_print=False)
                else:
                    qty = f.buy_stock(buy_price, 100, date=d)

                if qty > 0:
                    trade_log.append(
                        (
                            d,
                            "BUY",
                            buy_price,
                            qty,
                            f.get_total_amount(buy_price),
                            base_price,
                            real_t1l_price,
                            prob_t1l_buy,
                        )
                    )
                    have_position = True

        # 4) 每日权益按 base_price 估值
        total_equity = f.get_total_amount(base_price)
        equity_curve.append(total_equity)
        equity_dates.append(d)
        print(" 买入意向:", {True: "Yes", False: "No "}[has_buy_intent], f"T1L/T2H置信率:{prob_t1l_buy*100:.1f}%/{prob_t2h_sell*100:.1f}%, 买入量:[{int(qty):5d}], 当前权益: {total_equity:.2f}")

    # 回测结束，如仍有持仓，按最后一天收盘价强制平仓（可选）
    if f.get_stock_quantity() > 0 and len(equity_dates) > 0:
        last_date = equity_dates[-1]
        try:
            _, last_price = ds_t1l_buy.get_predictable_dataset_by_date(last_date)
            last_price = float(last_price)
        except Exception:
            last_price = base_price
        f.sell_all(last_price, date=last_date, is_print=False)
        total_equity = f.get_total_amount(last_price)
        equity_curve[-1] = total_equity

    final_equity = equity_curve[-1] if equity_curve else init_capital
    total_return = (final_equity - init_capital) / init_capital * 100

    print("=" * 80)
    print(f"回测股票: {stock_code}")
    print(f"模型: {model_type}")
    print(f"T1L_buy:  {t1l_buy_type} / {t1l_buy_feature}")
    print(f"T2H_sell:  {t2h_sell_type} / {t2h_sell_feature}")
    print(f"T1H_sell: {t1h_sell_type} / {t1h_sell_feature}")
    print(f"数据起点(内部实际使用): {ds_t1l_buy.stock.start_date}")
    print(f"回测区间(实际遍历): [{backtest_start}] - [{backtest_end}]")
    print(f"初始资金: {init_capital:.2f}")
    print(f"结束资金: {final_equity:.2f}")
    print(f"总收益率: {total_return:.2f}%")
    print(f"交易次数: {len(trade_log)}")
    print("=" * 80)

    for rec in trade_log:
        # rec: (d, side, price, qty, equity, base_price, real_T1x_price, prob)
        d, side, price, qty, equity, bp, real_px, prob = rec
        real_tag = "L" if side == "BUY" else "H"
        extra = f", T0_close={bp:.2f}, real_T1{real_tag}={real_px:.2f}, prob={prob:.3f}"
        print(f"{d} {side:4s} @ {price:.2f}, qty={qty}, equity={equity:.0f}{extra}")

    return {
        "dates": equity_dates,
        "equity": equity_curve,
        "trades": trade_log,
        "final_equity": final_equity,
        "total_return": total_return,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于 T1L/T1H/T2H 各自独立 Feature 的二分类模型，模拟挂单交易评估收益"
    )
    parser.add_argument("--stock_code", default="600036.SH", help="主股票代码")
    parser.add_argument(
        "--model_type",
        default="TRANSFORMER",
        choices=["RESIDUAL_LSTM", "RESIDUAL_TCN", "TRANSFORMER", "MINI", "CONV1D"],
        help="ModelType 枚举名",
    )

    # 买入信号使用的二分类 + 各自特征
    parser.add_argument(
        "--t1l_buy_type",
        default="BINARY_T1_L05",
        help="PredictType, 用于买入信号 + 买入挂单价，如 BINARY_T1_L05",
    )
    parser.add_argument(
        "--t1l_buy_feature",
        default="T1L05_F55",
        help="FeatureType, 对应 T1L 买入模型的特征，如 T1L10_F55",
    )
    parser.add_argument(
        "--t2h_sell_type",
        default="BINARY_T2_H10",
        help="PredictType, 用于买入信号，如 BINARY_T2_H10",
    )
    parser.add_argument(
        "--t2h_sell_feature",
        default="T2H10_F55",
        help="FeatureType, 对应 T2H 买入模型的特征，如 T2H10_F55",
    )

    # 卖出信号/挂单价使用的二分类 + 各自特征
    parser.add_argument(
        "--t1h_sell_type",
        default="BINARY_T1_H08",
        help="PredictType, 用于判断是否卖出 + 卖出挂单价（BINARY_T1_Hxx）",
    )
    parser.add_argument(
        "--t1h_sell_feature",
        default="T1H08_F55",
        help="FeatureType, 对应 T1H 卖出信号模型的特征，如 T1H10_F55",
    )

    parser.add_argument("--start_date", default="20250101", help="回测开始日期，YYYYMMDD")
    parser.add_argument("--end_date", default='20251115', help="回测结束日期，YYYYMMDD，默认到今天")
    parser.add_argument("--init_capital", type=float, default=500000, help="初始资金")
    parser.add_argument(
        "--no_buy_max",
        action="store_true",
        help="不满仓买入，每次只买一手（默认满仓 buy_max）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_type = ModelType.TRANSFORMER#getattr(ModelType, args.model_type.upper(), ModelType.TRANSFORMER)

    # PredictType/FeatureType（各自独立）
    t1l_buy_type = PredictType.BINARY_T1_L08#getattr(PredictType, args.t1l_buy_type, PredictType.BINARY_T1_L10)
    t1l_buy_feature = FeatureType.ALL#getattr(FeatureType, args.t1l_buy_feature.upper(), FeatureType.T1L10_F55)

    t2h_sell_type = PredictType.BINARY_T2_H10#getattr(PredictType, args.t2h_sell_type, PredictType.BINARY_T2_H10)
    t2h_sell_feature = FeatureType.T2H10_F55#getattr(FeatureType, args.t2h_sell_feature.upper(), FeatureType.T2H10_F55)

    t1h_sell_type = PredictType.BINARY_T1_H08#getattr(PredictType, args.t1h_sell_type, PredictType.BINARY_T1_H10)
    t1h_sell_feature = FeatureType.T1H08_F18#getattr(FeatureType, args.t1h_sell_feature.upper(), FeatureType.T1H10_F55)

    end_date = args.end_date or datetime.now().strftime("%Y%m%d")

    simulate_trading(
        stock_code=args.stock_code,
        model_type=model_type,
        t1l_buy_type=t1l_buy_type,
        t1l_buy_feature=t1l_buy_feature,
        t2h_sell_type=t2h_sell_type,
        t2h_sell_feature=t2h_sell_feature,
        t1h_sell_type=t1h_sell_type,
        t1h_sell_feature=t1h_sell_feature,
        start_date=args.start_date,
        end_date=end_date,
        init_capital=args.init_capital,
        use_buy_max=not args.no_buy_max,  # 默认 buy_max
    )