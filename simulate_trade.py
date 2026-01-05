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
from utils.utils import FeatureType, ModelType, PredictType, StrategyType, setup_logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _fmt_price(v):
    return "-----" if v is None else f"{float(v):.2f}"

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
    model_suffix: str = "",
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
    if model_suffix == "":
        model_sub_dir = ""
    else:
        model_sub_dir = f"{model_type}_{predict_type}"
        
    model = load_model_by_params(stock_code, model_type, predict_type, feature_type, suffix=model_suffix, sub_dir=model_sub_dir)
    return si, ds, model


def _predict_binary_label(model, x_input, threshold=0.5):
    """对二分类模型做预测, 返回 (prob, label)"""
    raw = model.model.predict(x_input, verbose=0)
    prob = float(raw[0, 0])
    label = int(prob > threshold)
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
    升序(旧->新)下：
      i0 = T0
      i1 = i0 + 1 = T1
    """
    i0 = _get_raw_index_by_date(ds, date_str)
    i1 = i0 + 1
    if i1 >= ds.raw_data.shape[0]:
        return None, None

    col_low = ds.p_trade.col_low + 1   # +1 是因为 raw_data 第0列是日期
    col_high = ds.p_trade.col_high + 1

    t1l = float(ds.raw_data[i1, col_low])
    t1h = float(ds.raw_data[i1, col_high])
    #print(f"日期:{date_str}, 真实 T1 价格: T1L={t1l}, T1H={t1h}")
    return t1l, t1h


def simulate_trading(
    stock_code: str,
    t1l_model_type: ModelType,
    t2h_model_type: ModelType,
    t1h_model_type: ModelType,
    # 买入信号用的二分类 + 特征
    t1l_pred_type: PredictType,      # BINARY_T1_Lxx
    t1l_feature_type: FeatureType,   # 如 FeatureType.T1L10_F55
    t2h_pred_type: PredictType,      # BINARY_T2_Hxx
    t2h_feature_type: FeatureType,   # 如 FeatureType.T2H10_F55
    # 卖出信号/挂单价用的二分类 + 特征
    t1h_pred_type: PredictType,     # BINARY_T1_Hzz （用来判断是否卖出 + 卖出挂单价）
    t1h_feature_type: FeatureType,  # 对应 T1H 的特征
    start_date: str,
    end_date: str,
    init_capital: float = 500000,
    use_buy_max: bool = True,
    t1l_threshold: float = 0.5,
    t2h_threshold: float = 0.5,
    t1h_threshold: float = 0.5,
    t1l_test_cyc: int = 1,
    t1h_test_cyc: int = 1,
    raise_threshold:float = 0.5,
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
    total_return_log = []   # 总收益率

    for cyc_t1l in range(t1l_test_cyc):
        for cyc_t1h in range(t1h_test_cyc):
            print(f"\n{'='*50} 回测周期 t1l:{cyc_t1l+1}/ t1h:{cyc_t1h+1} {'='*50}")   

            # ===== 构建四套 (feature, predict_type) 的数据集 & 模型 =====
            # 选 T1L 买入数据集 ds_t1l_buy 作为“主时间轴”，用它的 raw_data 取 T1L/T1H 真实价格
            si, ds_t1l, model_t1l = build_ds_and_model(
                stock_code=stock_code,
                feature_type=t1l_feature_type,
                model_type=t1l_model_type,
                predict_type=t1l_pred_type,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                train_size=1.0,
                model_suffix=str(cyc_t1l) if t1l_test_cyc > 1 else "",
            )
            _, ds_t1h, model_t1h = build_ds_and_model(
                stock_code=stock_code,
                feature_type=t1h_feature_type,
                model_type=t1h_model_type,
                predict_type=t1h_pred_type,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                train_size=1.0,
                model_suffix=str(cyc_t1h) if t1h_test_cyc > 1 else "",
            )
            _, ds_t2h, model_t2h = build_ds_and_model(
                stock_code=stock_code,
                feature_type=t2h_feature_type,
                model_type=t2h_model_type,
                predict_type=t2h_pred_type,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                train_size=1.0,
            )

            # ===== 回测日期（统一用 StockInfo 的交易日列表） =====
            trade_dates_df = si.get_trade_open_dates(backtest_start, backtest_end)
            date_list = trade_dates_df["trade_date"].astype(str).tolist()  # 从远到近

            f = Funds(init_amount=init_capital)

            equity_curve = []       # 每日权益曲线
            equity_dates = []       # 对应日期
            trade_log = []          # 交易记录列表
            have_position = False   # 当前是否有持仓

            for t0 in reversed(date_list):
                op_strategy = StrategyType.HOLD     # 当日策略, 默认不动
                is_op_success = False               # 当日是否有交易成功
                predict_price, real_price = 0,0     # 当日挂单价 & 真实价
                op_amount = 0                       # 当日操作数量  
                qty = 0                             # 当日实际成交数量
                label_t1l, label_t1h = 0,0
                #print("\n回测日期:", d)
                # ---- 针对 T0，在四个数据集上分别构造窗口 ----
                # 若任一关键数据集当天窗口不可用，则直接跳过该日
                try:
                    _, x_t1l, t0_close_price   = ds_t1l.get_predictable_dataset_by_date(t0)
                    _, x_t1h, _                = ds_t1h.get_predictable_dataset_by_date(t0)
                    _, x_t2h, _                = ds_t2h.get_predictable_dataset_by_date(t0)
                except Exception:
                    continue

                # 记账 & 真实价用 base_price：统一采用主 T1L 买入数据集的基准价（T0 收盘价）
                t0_close_price = float(t0_close_price)

                # ====================== 卖出逻辑 ======================
                if have_position and f.get_stock_quantity() > 0:    # 当前有持仓
                    op_strategy = StrategyType.SELL     # 当日先按卖出策略尝试卖出
                    op_amount = f.get_stock_quantity()  # 尝试卖出全部持仓
                    if t1h_pred_type.is_binary():
                        t1h_pred_val, label_t1h = _predict_binary_label(model_t1h, x_t1h, threshold=t1h_threshold)
                    elif t1h_pred_type.is_regression():
                        t1h_pred_val = float(model_t1h.model.predict(x_t1h, verbose=0)[0,0])
                        label_t1h = 1 # 总是有卖出意向
                    elif t1h_pred_type.is_classify():
                        #TODO: 多分类卖出意向判定
                        raise NotImplementedError("Classify type not implemented for T1H sell")
                    else:
                        raise ValueError(f"Unsupported PredictType for T1H sell: {t1h_pred_type}")

                    if label_t1h == 1:  #有卖出意向
                        if t1h_pred_type.is_binary():
                            sell_rate = t1h_pred_type.val / 100.0  # 正数，如 +1.0 -> 0.01
                        elif t1h_pred_type.is_regression():
                            sell_rate = t1h_pred_val / 100.0  # 按回归预测值卖出
                        elif t1h_pred_type.is_classify():
                            raise NotImplementedError("Classify type not implemented for T1H sell")
                        else:
                            raise ValueError(f"Unsupported PredictType for T1H sell: {t1h_pred_type}")
                    else:   #无卖出意向，按默认卖出底价卖出
                        sell_rate = SELL_RATE / 100  #按底价卖出

                    # 2) 卖出挂单价（由 BINARY_T1_Hzz 的阈值决定）
                    sell_price = t0_close_price * (1 + sell_rate)

                    # 3) 从“主 T1L 数据集”的 raw_data 中获取真实 T1H 价格
                    _, real_t1h_price = _get_real_t1_prices(ds_t1l, t0)
                    predict_price, real_price = sell_price, real_t1h_price

                    if real_t1h_price is not None and real_t1h_price >= sell_price:
                        qty = f.sell_all(sell_price, date=t0, is_print=False)
                        if qty > 0: # 卖出成功
                            is_op_success = True
                            have_position = False
                            trade_log.append(
                                (
                                    t0,
                                    "SELL",
                                    sell_price,
                                    qty,
                                    f.get_total_amount(sell_price),
                                    t0_close_price,
                                    real_t1h_price,
                                    t1h_pred_val,
                                )
                            )

                # ====================== 买入逻辑 ======================
                # 1) T1L/T2H 二分类给出买入意向（“T1低值跌，T2高值涨”）
                label_t1l_buy, label_t2h_sell = None, None
                if t1l_pred_type.is_binary():
                    t1l_pred_val, label_t1l_buy = _predict_binary_label(model_t1l, x_t1l, threshold=t1l_threshold)
                    t1l_pred_val = t1l_pred_type.val if label_t1l_buy == 1 else 0.0
                elif t1l_pred_type.is_regression():
                    t1l_pred_val = float(model_t1l.model.predict(x_t1l, verbose=0)[0,0])
                elif t1l_pred_type.is_classify():
                    raise NotImplementedError("Classify type not implemented for T1L buy")
                else:
                    raise ValueError(f"Unsupported PredictType for T1L buy: {t1l_pred_type}")
                
                if t2h_pred_type.is_binary():                    
                    t2h_pred_val, label_t2h_sell = _predict_binary_label(model_t2h, x_t2h, threshold=t2h_threshold)
                    t2h_pred_val = t2h_pred_type.val if label_t2h_sell == 1 else 0.0
                elif t2h_pred_type.is_regression():
                    t2h_pred_val = float(model_t2h.model.predict(x_t2h, verbose=0)[0,0])
                elif t2h_pred_type.is_classify():
                    raise NotImplementedError("Classify type not implemented for T2H sell")
                else:
                    raise ValueError(f"Unsupported PredictType for T2H sell: {t2h_pred_type}")

                has_buy_intent = (t2h_pred_val - t1l_pred_val) >= raise_threshold
                #print(f"t2h - t1l = {t2h_pred_val - t1l_pred_val:.2f} , t2h:{t2h_pred_val:.2f} , t1l:{t1l_pred_val:.2f}")
                if has_buy_intent: #有买入意向
                    if t1l_pred_type.is_binary():
                        buy_rate = t1l_pred_type.val / 100.0 
                    elif t1l_pred_type.is_regression():
                        buy_rate = t1l_pred_val / 100.0
                    elif t1l_pred_type.is_classify():
                        raise NotImplementedError("Classify type not implemented for T1L buy")
                    else:
                        raise ValueError(f"Unsupported PredictType for T1L buy: {t1l_pred_type}")
                else:   #无买入意向，按默认买入底价买入
                    buy_rate = BUY_RATE / 100.0 #无买入意向，按底价买入

                if has_buy_intent and f.get_stock_quantity() == 0 and op_strategy != StrategyType.SELL: # 当前无持仓，且当天未按卖出策略卖出
                    op_strategy = StrategyType.BUY
                    # 2) 买入挂单价
                    buy_price = t0_close_price * (1 + buy_rate)
                    op_amount = f.get_buy_max_quantity(buy_price)

                    # 3) 真实 T1L 价格来自“主 T1L 数据集”的 raw_data T1 日 low
                    real_t1l_price, _ = _get_real_t1_prices(ds_t1l, t0)
                    predict_price, real_price = buy_price, real_t1l_price

                    if real_t1l_price is not None and real_t1l_price <= buy_price:
                        # 挂单成交
                        if use_buy_max:
                            qty = f.buy_max(buy_price, date=t0, is_print=False)
                        else:
                            qty = f.buy_stock(buy_price, 100, date=t0)

                        if qty > 0:
                            is_op_success = True
                            have_position = True
                            #print(f"买入成功, 当前持仓:{f.get_stock_quantity()}")
                            trade_log.append(
                                (
                                    t0,
                                    "BUY",
                                    buy_price,
                                    qty,
                                    f.get_total_amount(buy_price),
                                    t0_close_price,
                                    real_t1l_price,
                                    t1l_pred_val,
                                )
                            )

                # 4) 每日权益按 base_price 估值
                total_equity = f.get_total_amount(t0_close_price)
                equity_curve.append(total_equity)
                equity_dates.append(t0)

                str_buy_intent = "Yes" if has_buy_intent else "No " #买入意向
                str_confidence = f"T1L/T2H置信率:{t1l_pred_val*100:.1f}%/{t2h_pred_val*100:.1f}%" if t1l_pred_type.is_binary() and t2h_pred_type.is_binary() else ""
                str_base_price = f"{t0_close_price:.2f}"
                str_op_strategy = f"{op_strategy}"
                #str_buy_judge = f"({label_t1l_buy}/{label_t2h_sell})" if label_t1l_buy is not None and label_t2h_sell is not None else f"({(t2h_pred_val - t1l_pred_val): .1f})"
                str_buy_judge = f"({t2h_pred_val: .1f}/{t1l_pred_val: .1f})"
                #str_buy_judge = f"({(t2h_pred_val - t1l_pred_val): .1f})"
                str_op_price = f"{_fmt_price(predict_price)}/({_fmt_price(real_price)})" if op_strategy != StrategyType.HOLD else "             "
                str_op_amount = f"{int(op_amount):5d}" if op_strategy != StrategyType.HOLD else "-----"
                str_is_op_success = f"{' OK.' if is_op_success else 'NOK!'}" if op_strategy != StrategyType.HOLD else "    "
                str_cur_qty = f"{int(f.get_stock_quantity()):5d}"
                str_equity = f"{total_equity:.2f}"
                print(f"{t0}({str_base_price}) : <{str_op_strategy}{str_buy_judge}> [{str_op_amount}]@[{str_op_price}] - {str_is_op_success}, 当前持仓/资本:{str_cur_qty}/{str_equity}")

            # 回测结束，如仍有持仓，按最后一天收盘价强制平仓（可选）
            if f.get_stock_quantity() > 0 and len(equity_dates) > 0:
                last_date = equity_dates[-1]
                try:
                    _, _, last_price = ds_t1l.get_predictable_dataset_by_date(last_date)
                    last_price = float(last_price)
                except Exception:
                    last_price = t0_close_price
                f.sell_all(last_price, date=last_date, is_print=False)
                total_equity = f.get_total_amount(last_price)
                equity_curve[-1] = total_equity

            final_equity = equity_curve[-1] if equity_curve else init_capital
            total_return = (final_equity - init_capital) / init_capital * 100
            total_return_log.append(f"t1l cyc:{cyc_t1l} , t1h cyc:{cyc_t1h} , return:{total_return:.2f}") 

            # --- 回测区间涨跌幅：直接用 raw_data 的 T0 close 计算，不走窗口接口 ---
            start_date_bt = date_list[-1]    # 回测起点(最早)
            end_date_bt   = date_list[0]   # 回测终点(最晚)

            col_close = ds_t1l.p_trade.col_close + 1  # raw_data 第0列是日期

            start_idx = _get_raw_index_by_date(ds_t1l, start_date_bt)
            end_idx   = _get_raw_index_by_date(ds_t1l, end_date_bt)

            start_price = float(ds_t1l.raw_data[start_idx, col_close])
            end_price   = float(ds_t1l.raw_data[end_idx, col_close])
            
            print("=" * 90)
            print(f"")
            print(f"回测股票: {stock_code}")
            print(f"模型(t1l/t2h/t1h): {t1l_model_type}/{t2h_model_type}/{t1h_model_type}")
            print(f"T1L_buy:  {t1l_pred_type} / {t1l_feature_type}")
            print(f"T2H_sell: {t2h_pred_type} / {t2h_feature_type}")
            print(f"T1H_sell: {t1h_pred_type} / {t1h_feature_type}")
            print(f"数据起点(内部实际使用): {ds_t1l.stock.start_date}")
            print(f"回测区间(实际遍历)/实际涨跌幅: [{backtest_start}/({start_price})] - [{backtest_end}/({end_price})]/{100*(end_price-start_price)/start_price:.2f}%")
            print(f"初始资金: {init_capital:.2f}")
            print(f"结束资金: {final_equity:.2f}")
            print(f"总收益率: {total_return:.2f}%")
            print(f"交易次数: {len(trade_log)}")
            print("=" * 90)

            for rec in trade_log:
                # rec: (d, side, price, qty, equity, base_price, real_T1x_price, prob)
                t0, side, price, qty, equity, bp, real_px, prob = rec
                real_tag = "L" if side == "BUY" else "H"
                extra = f", T0_close={bp:.2f}, real_T1{real_tag}={real_px:.2f}, prob={prob:.3f}"
                #print(f"{t0} {side:4s} @ {price:.2f}, qty={qty}, equity={equity:.0f}{extra}")

    return {
        "dates": equity_dates,
        "equity": equity_curve,
        "trades": trade_log,
        "final_equity": final_equity,
        "total_return": total_return,
        "total_return_log": total_return_log,
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
    BUY_RATE, SELL_RATE = -0.49, 0.49
    RAISE_THRESHOLD = 2   #[15.83%:0.6]
    T1L_TEST_CYC, T1H_TEST_CYC = 1, 1
    #T1L_TEST_CYC, T1H_TEST_CYC = 1, 1

    #start_date, end_date = args.start_date, args.end_date
    start_date_down, end_date_down = '20250701', '20250825'#下降周期
    start_date_normal_up, end_date_normal_up = '20250715', '20251229'#波动周期(总体上涨)

    # PredictType/FeatureType（各自独立）
    t1l_model_type = ModelType.RESIDUAL_LSTM#getattr(ModelType, args.model_type.upper(), ModelType.TRANSFORMER)
    t1l_buy_type = PredictType.BINARY_T1_L10#getattr(PredictType, args.t1l_buy_type, PredictType.BINARY_T1_L10), REGRESS_T1L
    t1l_buy_feature = FeatureType.BINARY_T1L10_F55#getattr(FeatureType, args.t1l_buy_feature.upper(), FeatureType.T1L10_F55), REGRESS_T1L_F50
    t1l_th = 0.477

    t1h_model_type = ModelType.RESIDUAL_LSTM#getattr(ModelType, args.model_type.upper(), ModelType.TRANSFORMER)
    t1h_sell_type = PredictType.BINARY_T1_H10#getattr(PredictType, args.t1h_sell_type, PredictType.BINARY_T1_H10)
    t1h_sell_feature = FeatureType.BINARY_T1H10_F55#getattr(FeatureType, args.t1h_sell_feature.upper(), FeatureType.T1H10_F55)
    t1h_th = 0.624

    t2h_model_type = ModelType.RESIDUAL_LSTM#getattr(ModelType, args.model_type.upper(), ModelType.TRANSFORMER)
    t2h_sell_type = PredictType.BINARY_T2_H10#REGRESS_T2H#getattr(PredictType, args.t2h_sell_type, PredictType.BINARY_T2_H10)
    t2h_sell_feature = FeatureType.BINARY_T2H10_F25#getattr(FeatureType, args.t2h_sell_feature.upper(), FeatureType.T2H10_F55)
    t2h_th = 0.436

    #end_date = args.end_date or datetime.now().strftime("%Y%m%d")

    normal_up_result = simulate_trading(
        stock_code=args.stock_code,
        t1l_model_type=t1l_model_type,
        t2h_model_type=t2h_model_type,
        t1h_model_type=t1h_model_type,
        t1l_pred_type=t1l_buy_type,
        t1l_feature_type=t1l_buy_feature,
        t2h_pred_type=t2h_sell_type,
        t2h_feature_type=t2h_sell_feature,
        t1h_pred_type=t1h_sell_type,
        t1h_feature_type=t1h_sell_feature,
        start_date=start_date_normal_up,
        end_date=end_date_normal_up,
        init_capital=args.init_capital,
        use_buy_max=not args.no_buy_max,  # 默认 buy_max
        t1l_threshold=t1l_th,
        t2h_threshold=t2h_th,
        t1h_threshold=t1h_th,
        t1l_test_cyc = T1L_TEST_CYC,
        t1h_test_cyc = T1H_TEST_CYC,
        raise_threshold=RAISE_THRESHOLD,
    )

    down_result = simulate_trading(
        stock_code=args.stock_code,
        t1l_model_type=t1l_model_type,
        t2h_model_type=t2h_model_type,
        t1h_model_type=t1h_model_type,
        t1l_pred_type=t1l_buy_type,
        t1l_feature_type=t1l_buy_feature,
        t2h_pred_type=t2h_sell_type,
        t2h_feature_type=t2h_sell_feature,
        t1h_pred_type=t1h_sell_type,
        t1h_feature_type=t1h_sell_feature,
        start_date=start_date_down,
        end_date=end_date_down,
        init_capital=args.init_capital,
        use_buy_max=not args.no_buy_max,  # 默认 buy_max
        t1l_threshold=t1l_th,
        t2h_threshold=t2h_th,
        t1h_threshold=t1h_th,
        t1l_test_cyc = T1L_TEST_CYC,
        t1h_test_cyc = T1H_TEST_CYC,
        raise_threshold=RAISE_THRESHOLD,
    )

    for cyc_idx, ret in enumerate(normal_up_result["total_return_log"]):
        print(f"波动周期(总体上涨), 回测收益情况: {ret}")

    for cyc_idx, ret in enumerate(down_result["total_return_log"]):
        print(f"下降周期, 回测收益情况: {ret}")        

