# -*- coding: utf-8 -*-
"""
单标的回测脚本 - 600036.SH 2025年纯OOS回测
使用二分类模型(BINARY_T1L10)进行0/1择时策略回测，并与Buy & Hold对比

运行方式:
    python backtest_single_600036_2025.py

输出:
    - 控制台：2024参数选择结果、2025回测指标对比
    - CSV文件：策略和基准的净值曲线保存到 data/temp/
"""
import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.stockinfo import StockInfo
from datasets.dataset import StockDataset
from datasets.money import Funds
from model.utils import load_model_by_params
from backtest.backtest_utils import (
    get_next_day_open_price,
    get_close_price,
    calculate_max_drawdown,
    calculate_trade_metrics,
    save_backtest_results,
    print_backtest_summary
)
from utils.tk import TOKEN
from utils.const_def import IDX_CODE_LIST, CONTINUOUS_DAYS, BASE_DIR, TMP_DIR
from utils.utils import FeatureType, ModelType, PredictType, setup_logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def select_thresholds_on_2024(
    ds: StockDataset,
    model,
    trade_dates_2024: list,
    initial_capital: float = 100000,
    thr_buy_range: list = None,
    thr_sell_range: list = None
):
    """
    在2024年数据上扫描阈值对，选择最优参数
    
    Args:
        ds: 数据集
        model: 训练好的模型
        trade_dates_2024: 2024年的交易日列表
        initial_capital: 初始资金
        thr_buy_range: thr_buy候选值列表
        thr_sell_range: thr_sell候选值列表
        
    Returns:
        (best_thr_buy, best_thr_sell, best_return, best_mdd)
    """
    if thr_buy_range is None:
        # 保守起见，选择较低的买入阈值（更容易触发买入）
        thr_buy_range = [0.1, 0.15, 0.2, 0.25, 0.3]
    if thr_sell_range is None:
        # 卖出阈值应该大于买入阈值，形成滞回
        thr_sell_range = [0.5, 0.55, 0.6, 0.65, 0.7]
    
    print("\n" + "="*80)
    print("开始在2024年数据上选择阈值...")
    print("="*80)
    
    best_return = -np.inf
    best_thr_buy = None
    best_thr_sell = None
    best_mdd = 0
    
    results = []
    
    for thr_buy in thr_buy_range:
        for thr_sell in thr_sell_range:
            if thr_buy >= thr_sell:
                continue  # 必须 thr_buy < thr_sell
            
            # 运行策略
            funds = Funds(initial_capital)
            position = 0  # 0=空仓, 1=持仓
            equity_curve = []
            
            for date_str in trade_dates_2024:
                try:
                    # 获取T0收盘后的模型输入
                    _, x, close = ds.get_predictable_dataset_by_date(date_str)
                    
                    # 模型预测
                    raw = model.model.predict(x, verbose=0)
                    prob = float(raw[0, 0])
                    
                    # 获取次日开盘价
                    next_open = get_next_day_open_price(ds, date_str)
                    if next_open is None:
                        # 无次日数据，记录当前净值
                        equity = funds.get_total_amount(close)
                        equity_curve.append(equity)
                        continue
                    
                    # 生成信号并执行交易
                    if prob <= thr_buy and position == 0:
                        # 买入信号
                        funds.buy_max(next_open, date=date_str, is_print=False)
                        position = 1
                    elif prob >= thr_sell and position == 1:
                        # 卖出信号
                        funds.sell_all(next_open, date=date_str, is_print=False)
                        position = 0
                    
                    # 记录净值
                    equity = funds.get_total_amount(next_open)
                    equity_curve.append(equity)
                    
                except Exception as e:
                    # 某些日期可能不可预测（数据不足等），跳过
                    continue
            
            if len(equity_curve) == 0:
                continue
            
            # 计算指标
            final_equity = equity_curve[-1]
            total_return = (final_equity - initial_capital) / initial_capital
            mdd, _, _ = calculate_max_drawdown(equity_curve)
            
            results.append({
                'thr_buy': thr_buy,
                'thr_sell': thr_sell,
                'return': total_return,
                'mdd': mdd,
                'final_equity': final_equity
            })
            
            # 更新最佳参数（以收益为主）
            if total_return > best_return:
                best_return = total_return
                best_thr_buy = thr_buy
                best_thr_sell = thr_sell
                best_mdd = mdd
    
    # 打印前5个结果
    print("\n2024年参数扫描结果（按收益率排序，前5名）:")
    print("-"*80)
    print(f"{'thr_buy':<10} {'thr_sell':<10} {'收益率':<15} {'最大回撤':<15}")
    print("-"*80)
    
    results_sorted = sorted(results, key=lambda x: x['return'], reverse=True)
    for i, r in enumerate(results_sorted[:5]):
        print(f"{r['thr_buy']:<10.2f} {r['thr_sell']:<10.2f} {r['return']*100:<15.2f}% {r['mdd']*100:<15.2f}%")
    
    print("\n选定参数:")
    print(f"thr_buy={best_thr_buy}, thr_sell={best_thr_sell}")
    print(f"2024年收益率: {best_return*100:.2f}%, 最大回撤: {best_mdd*100:.2f}%")
    print("="*80 + "\n")
    
    return best_thr_buy, best_thr_sell, best_return, best_mdd


def run_strategy_backtest(
    ds: StockDataset,
    model,
    trade_dates: list,
    thr_buy: float,
    thr_sell: float,
    initial_capital: float = 100000
):
    """
    运行策略回测
    
    Args:
        ds: 数据集
        model: 训练好的模型
        trade_dates: 交易日列表
        thr_buy: 买入阈值
        thr_sell: 卖出阈值
        initial_capital: 初始资金
        
    Returns:
        (equity_curve, trades, dates_recorded, positions)
    """
    funds = Funds(initial_capital)
    position = 0  # 0=空仓, 1=持仓
    equity_curve = []
    dates_recorded = []
    positions = []
    trades = []
    
    current_trade = None  # 记录当前交易
    
    for date_str in trade_dates:
        try:
            # 获取T0收盘后的模型输入
            _, x, close = ds.get_predictable_dataset_by_date(date_str)
            
            # 模型预测
            raw = model.model.predict(x, verbose=0)
            prob = float(raw[0, 0])
            
            # 获取次日开盘价
            next_open = get_next_day_open_price(ds, date_str)
            if next_open is None:
                # 无次日数据，使用收盘价记录净值
                equity = funds.get_total_amount(close)
                equity_curve.append(equity)
                dates_recorded.append(date_str)
                positions.append(funds.get_stock_quantity())
                continue
            
            # 生成信号并执行交易
            if prob <= thr_buy and position == 0:
                # 买入信号
                qty = funds.buy_max(next_open, date=date_str, is_print=False)
                if qty > 0:
                    position = 1
                    current_trade = {
                        'buy_date': date_str,
                        'buy_price': next_open,
                        'buy_cost': funds.cur_buy_cost
                    }
            elif prob >= thr_sell and position == 1:
                # 卖出信号
                qty = funds.sell_all(next_open, date=date_str, is_print=False)
                if qty > 0:
                    position = 0
                    if current_trade is not None:
                        # 计算收益
                        sell_revenue = qty * next_open * (1 - funds.sell_fee)
                        profit = sell_revenue - current_trade['buy_cost']
                        ret = profit / current_trade['buy_cost']
                        
                        trades.append({
                            'buy_date': current_trade['buy_date'],
                            'buy_price': current_trade['buy_price'],
                            'sell_date': date_str,
                            'sell_price': next_open,
                            'profit': profit,
                            'return': ret
                        })
                        current_trade = None
            
            # 记录净值
            equity = funds.get_total_amount(next_open)
            equity_curve.append(equity)
            dates_recorded.append(date_str)
            positions.append(funds.get_stock_quantity())
            
        except Exception as e:
            # 某些日期可能不可预测，跳过
            continue
    
    return equity_curve, trades, dates_recorded, positions


def run_buy_and_hold(
    ds: StockDataset,
    trade_dates: list,
    initial_capital: float = 100000
):
    """
    运行Buy & Hold基准策略
    在第一个可交易日买入并持有至结束
    
    Args:
        ds: 数据集
        trade_dates: 交易日列表
        initial_capital: 初始资金
        
    Returns:
        (equity_curve, trades, dates_recorded, positions)
    """
    funds = Funds(initial_capital)
    equity_curve = []
    dates_recorded = []
    positions = []
    trades = []
    bought = False
    buy_date = None
    buy_price = None
    buy_cost = None
    
    for date_str in trade_dates:
        try:
            close = get_close_price(ds, date_str)
            if close is None:
                continue
            
            # 获取次日开盘价用于买入
            next_open = get_next_day_open_price(ds, date_str)
            
            if not bought and next_open is not None:
                # 在第一个有次日开盘价的日期买入
                qty = funds.buy_max(next_open, date=date_str, is_print=False)
                if qty > 0:
                    bought = True
                    buy_date = date_str
                    buy_price = next_open
                    buy_cost = funds.cur_buy_cost
            
            # 记录净值
            if next_open is not None:
                equity = funds.get_total_amount(next_open)
            else:
                equity = funds.get_total_amount(close)
            
            equity_curve.append(equity)
            dates_recorded.append(date_str)
            positions.append(funds.get_stock_quantity())
            
        except Exception as e:
            continue
    
    # 在最后一天卖出（仅用于计算交易指标）
    if bought and len(dates_recorded) > 0:
        last_date = dates_recorded[-1]
        last_price = equity_curve[-1] / funds.get_stock_quantity() if funds.get_stock_quantity() > 0 else 0
        
        if last_price > 0:
            qty = funds.get_stock_quantity()
            sell_revenue = qty * last_price * (1 - funds.sell_fee)
            profit = sell_revenue - buy_cost
            ret = profit / buy_cost
            
            trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': last_date,
                'sell_price': last_price,
                'profit': profit,
                'return': ret
            })
    
    return equity_curve, trades, dates_recorded, positions


def main():
    """主函数"""
    setup_logging()
    
    # ========== 配置参数 ==========
    STOCK_CODE = '600036.SH'
    MODEL_TYPE = ModelType.TRANSFORMER
    FEATURE_TYPE = FeatureType.BINARY_T1L10_F55
    PREDICT_TYPE = PredictType.BINARY_T1L10
    INITIAL_CAPITAL = 100000
    
    # 数据区间
    DATA_START = '20230101'  # 数据起始（需要足够历史数据用于窗口）
    YEAR_2024_START = '20240102'
    YEAR_2024_END = '20241231'
    YEAR_2025_START = '20250102'
    YEAR_2025_END = '20251231'
    
    print("\n" + "="*80)
    print(f"回测脚本 - {STOCK_CODE} 2025年纯OOS测试")
    print("="*80)
    print(f"股票代码: {STOCK_CODE}")
    print(f"模型类型: {MODEL_TYPE}")
    print(f"特征类型: {FEATURE_TYPE}")
    print(f"预测类型: {PREDICT_TYPE}")
    print(f"初始资金: ¥{INITIAL_CAPITAL:,.2f}")
    print("="*80 + "\n")
    
    # ========== 加载数据和模型 ==========
    print("正在加载数据和模型...")
    si = StockInfo(TOKEN)
    
    # 构建数据集（包含2024和2025数据）
    ds = StockDataset(
        ts_code=STOCK_CODE,
        idx_code_list=[],  # 不使用指数
        rel_code_list=[],  # 不使用关联股票
        si=si,
        start_date=DATA_START,
        end_date=YEAR_2025_END,
        train_size=1.0,  # 不分训练集，全部用于回测
        feature_type=FEATURE_TYPE,
        predict_type=PREDICT_TYPE,
        if_update_scaler=False
    )
    
    # 加载模型（假设模型已训练好）
    try:
        model = load_model_by_params(
            STOCK_CODE,
            MODEL_TYPE,
            PREDICT_TYPE,
            FEATURE_TYPE,
            suffix="",  # 根据实际模型文件名调整
            sub_dir=""
        )
        print("模型加载成功\n")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已训练好对应的模型文件")
        return
    
    # 获取交易日
    all_dates = ds.raw_data[:, 0].astype(str).tolist()
    trade_dates_2024 = [d for d in all_dates if YEAR_2024_START <= d <= YEAR_2024_END]
    trade_dates_2025 = [d for d in all_dates if YEAR_2025_START <= d <= YEAR_2025_END]
    
    print(f"2024年交易日数量: {len(trade_dates_2024)}")
    print(f"2025年交易日数量: {len(trade_dates_2025)}\n")
    
    # ========== 第一步：在2024年选择阈值 ==========
    thr_buy, thr_sell, ret_2024, mdd_2024 = select_thresholds_on_2024(
        ds, model, trade_dates_2024, INITIAL_CAPITAL
    )
    
    # ========== 第二步：在2025年运行策略回测 ==========
    print("\n" + "="*80)
    print("开始2025年OOS回测...")
    print("="*80 + "\n")
    
    strategy_equity, strategy_trades, strategy_dates, strategy_positions = run_strategy_backtest(
        ds, model, trade_dates_2025, thr_buy, thr_sell, INITIAL_CAPITAL
    )
    
    # ========== 第三步：运行Buy & Hold基准 ==========
    print("运行Buy & Hold基准策略...\n")
    bh_equity, bh_trades, bh_dates, bh_positions = run_buy_and_hold(
        ds, trade_dates_2025, INITIAL_CAPITAL
    )
    
    # ========== 第四步：计算并输出指标 ==========
    # 策略指标
    strategy_final = strategy_equity[-1] if strategy_equity else INITIAL_CAPITAL
    strategy_mdd, _, _ = calculate_max_drawdown(strategy_equity)
    strategy_metrics = calculate_trade_metrics(strategy_trades)
    
    # 基准指标
    bh_final = bh_equity[-1] if bh_equity else INITIAL_CAPITAL
    bh_mdd, _, _ = calculate_max_drawdown(bh_equity)
    bh_metrics = calculate_trade_metrics(bh_trades)
    
    # 打印摘要
    print_backtest_summary(
        "择时策略",
        INITIAL_CAPITAL,
        strategy_final,
        strategy_mdd,
        strategy_metrics,
        YEAR_2025_START,
        YEAR_2025_END
    )
    
    print_backtest_summary(
        "Buy & Hold",
        INITIAL_CAPITAL,
        bh_final,
        bh_mdd,
        bh_metrics,
        YEAR_2025_START,
        YEAR_2025_END
    )
    
    # ========== 第五步：保存结果 ==========
    output_dir = os.path.join(BASE_DIR, TMP_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存策略净值
    strategy_file = os.path.join(output_dir, f"backtest_strategy_2025_{timestamp}.csv")
    save_backtest_results(
        strategy_file,
        strategy_dates,
        strategy_equity,
        strategy_positions
    )
    
    # 保存基准净值
    bh_file = os.path.join(output_dir, f"backtest_buyhold_2025_{timestamp}.csv")
    save_backtest_results(
        bh_file,
        bh_dates,
        bh_equity,
        bh_positions
    )
    
    # 保存对比结果
    comparison_file = os.path.join(output_dir, f"backtest_comparison_2025_{timestamp}.txt")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write(f"回测对比 - {STOCK_CODE} 2025年\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"参数选择（2024年）:\n")
        f.write(f"  thr_buy={thr_buy}, thr_sell={thr_sell}\n")
        f.write(f"  2024收益率: {ret_2024*100:.2f}%, 最大回撤: {mdd_2024*100:.2f}%\n\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"2025年OOS测试结果:\n\n")
        f.write(f"策略表现:\n")
        f.write(f"  累计收益: {(strategy_final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:+.2f}%\n")
        f.write(f"  最大回撤: {strategy_mdd*100:.2f}%\n")
        f.write(f"  交易次数: {strategy_metrics['total_trades']}\n")
        f.write(f"  胜率: {strategy_metrics['win_rate']*100:.2f}%\n")
        f.write(f"  盈亏比: {strategy_metrics['profit_loss_ratio']:.2f}\n")
        f.write(f"  盈利因子: {strategy_metrics['profit_factor']:.2f}\n\n")
        f.write(f"Buy & Hold表现:\n")
        f.write(f"  累计收益: {(bh_final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100:+.2f}%\n")
        f.write(f"  最大回撤: {bh_mdd*100:.2f}%\n")
        f.write(f"  交易次数: {bh_metrics['total_trades']}\n\n")
    
    print(f"对比结果已保存到: {comparison_file}\n")
    
    print("\n" + "="*80)
    print("回测完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
