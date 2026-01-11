# -*- coding: utf-8 -*-
"""
回测工具函数模块
提供回测所需的各种辅助函数：获取价格、计算指标、保存结果等
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datasets.dataset import StockDataset


def get_next_day_open_price(ds: StockDataset, date_str: str) -> float:
    """
    获取指定日期的次日开盘价
    
    Args:
        ds: StockDataset实例
        date_str: 交易日期，格式如 '20250102'
        
    Returns:
        次日开盘价，如果次日不存在则返回None
    """
    # raw_data的第0列是日期
    date_val = type(ds.raw_data[0, 0])(date_str)
    idx_arr = np.where(ds.raw_data[:, 0] == date_val)[0]
    
    if idx_arr.size == 0:
        return None
        
    idx = int(idx_arr[0])
    # 次日索引
    next_idx = idx + 1
    
    if next_idx >= ds.raw_data.shape[0]:
        return None
        
    # raw_data第0列是日期，后面的列才是价格特征
    # col_open 是在 trade_df 中的列索引，在 raw_data 中需要+1（因为raw_data多了一个日期列）
    next_open = float(ds.raw_data[next_idx, ds.p_trade.col_open + 1])
    return next_open


def get_close_price(ds: StockDataset, date_str: str) -> float:
    """
    获取指定日期的收盘价
    
    Args:
        ds: StockDataset实例
        date_str: 交易日期，格式如 '20250102'
        
    Returns:
        收盘价，如果不存在则返回None
    """
    date_val = type(ds.raw_data[0, 0])(date_str)
    idx_arr = np.where(ds.raw_data[:, 0] == date_val)[0]
    
    if idx_arr.size == 0:
        return None
        
    idx = int(idx_arr[0])
    close = float(ds.raw_data[idx, ds.p_trade.col_close + 1])
    return close


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    """
    计算最大回撤及其起止位置
    
    Args:
        equity_curve: 净值曲线列表
        
    Returns:
        (max_drawdown, start_idx, end_idx)
        max_drawdown: 最大回撤百分比（负数）
        start_idx: 回撤开始位置
        end_idx: 回撤结束位置（最低点）
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0
        
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    
    max_dd_idx = np.argmin(drawdown)
    max_dd = float(drawdown[max_dd_idx])
    
    # 找到回撤开始位置（最高点）
    start_idx = np.argmax(running_max[:max_dd_idx+1])
    
    return max_dd, start_idx, max_dd_idx


def calculate_trade_metrics(trades: List[Dict]) -> Dict:
    """
    计算交易统计指标
    
    Args:
        trades: 交易记录列表，每条记录包含：
            {
                'buy_date': 买入日期,
                'buy_price': 买入价格,
                'sell_date': 卖出日期,
                'sell_price': 卖出价格,
                'profit': 收益（考虑手续费后）,
                'return': 收益率
            }
            
    Returns:
        包含交易指标的字典：
        {
            'total_trades': 总交易次数,
            'win_trades': 盈利交易次数,
            'lose_trades': 亏损交易次数,
            'win_rate': 胜率,
            'avg_win': 平均盈利,
            'avg_loss': 平均亏损,
            'profit_loss_ratio': 盈亏比,
            'profit_factor': 盈利因子（总盈利/总亏损）
        }
    """
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_trades': 0,
            'lose_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_loss_ratio': 0.0,
            'profit_factor': 0.0
        }
    
    total_trades = len(trades)
    wins = [t for t in trades if t['profit'] > 0]
    losses = [t for t in trades if t['profit'] <= 0]
    
    win_trades = len(wins)
    lose_trades = len(losses)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    avg_win = np.mean([t['profit'] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t['profit'] for t in losses]) if losses else 0.0
    
    # 盈亏比（平均盈利 / |平均亏损|）
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    
    # 盈利因子（总盈利 / |总亏损|）
    total_win = sum([t['profit'] for t in wins])
    total_loss = abs(sum([t['profit'] for t in losses]))
    profit_factor = total_win / total_loss if total_loss > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_trades': win_trades,
        'lose_trades': lose_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'profit_factor': profit_factor
    }


def save_backtest_results(
    filepath: str,
    dates: List[str],
    equity_curve: List[float],
    positions: List[int],
    signals: List[int] = None
):
    """
    保存回测结果到CSV文件
    
    Args:
        filepath: 保存路径
        dates: 日期列表
        equity_curve: 净值曲线
        positions: 持仓状态（0或持仓数量）
        signals: 信号列表（可选）
    """
    df = pd.DataFrame({
        'date': dates,
        'equity': equity_curve,
        'position': positions
    })
    
    if signals is not None:
        df['signal'] = signals
    
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"回测结果已保存到: {filepath}")


def print_backtest_summary(
    name: str,
    initial_capital: float,
    final_equity: float,
    max_drawdown: float,
    trade_metrics: Dict,
    start_date: str,
    end_date: str
):
    """
    打印回测摘要
    
    Args:
        name: 策略名称
        initial_capital: 初始资金
        final_equity: 最终净值
        max_drawdown: 最大回撤
        trade_metrics: 交易指标字典
        start_date: 回测开始日期
        end_date: 回测结束日期
    """
    total_return = (final_equity - initial_capital) / initial_capital
    
    print(f"\n{'='*60}")
    print(f"策略: {name}")
    print(f"回测区间: {start_date} - {end_date}")
    print(f"{'='*60}")
    print(f"初始资金: ¥{initial_capital:,.2f}")
    print(f"最终净值: ¥{final_equity:,.2f}")
    print(f"累计收益: {total_return*100:+.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print(f"-"*60)
    print(f"总交易次数: {trade_metrics['total_trades']}")
    print(f"盈利次数: {trade_metrics['win_trades']}")
    print(f"亏损次数: {trade_metrics['lose_trades']}")
    print(f"胜率: {trade_metrics['win_rate']*100:.2f}%")
    print(f"平均盈利: ¥{trade_metrics['avg_win']:,.2f}")
    print(f"平均亏损: ¥{trade_metrics['avg_loss']:,.2f}")
    print(f"盈亏比: {trade_metrics['profit_loss_ratio']:.2f}")
    print(f"盈利因子: {trade_metrics['profit_factor']:.2f}")
    print(f"{'='*60}\n")
