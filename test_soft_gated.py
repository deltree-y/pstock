#!/usr/bin/env python
# coding=utf-8
"""
测试软门控预测功能的示例脚本

使用方法：
    python test_soft_gated.py --stock_code 600036.SH --dates 20230601 20230602 20230605

注意：
    - 需要预先训练好 BINARY_T1L10 和 REGRESS_T1L 模型
    - 需要在 utils/tk.py 中配置 TuShare API Token
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.stockinfo import StockInfo
from datasets.dataset import StockDataset
from model.utils import load_model_by_params
from utils.tk import TOKEN
from utils.const_def import IDX_CODE_LIST
from utils.utils import FeatureType, ModelType, PredictType, setup_logging
from predicproc.show import print_predict_result_soft_gated_t1l10


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser(description="测试软门控预测功能")
    parser.add_argument("--stock_code", default="600036.SH", help="股票代码")
    parser.add_argument("--dates", nargs="+", required=False, help="预测日期列表，格式: YYYYMMDD")
    parser.add_argument("--from_date", default=None, help="如果设置，预测从此日期到今天的所有交易日")
    parser.add_argument("--start_date", default="20190104", help="数据起始日期")
    parser.add_argument("--end_date", default=None, help="数据结束日期")
    parser.add_argument("--y_base", type=float, default=-0.2, help="基础回归值（百分点），默认 -0.2")
    parser.add_argument("--gamma", type=float, default=1.0, help="锐化参数，默认 1.0")
    parser.add_argument("--model_type", default="TRANSFORMER", choices=["RESIDUAL_LSTM", "RESIDUAL_TCN", "TRANSFORMER", "CONV1D"],
                        help="模型类型")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # 设置模型类型和特征类型
    model_type = ModelType[args.model_type]
    gate_feature_type = FeatureType.BINARY_T1L10_F55  # 二分类门控模型
    reg_feature_type = FeatureType.REGRESS_T1L_F55    # 回归模型
    
    gate_predict_type = PredictType.get_type_from_feature_type(gate_feature_type)
    reg_predict_type = PredictType.get_type_from_feature_type(reg_feature_type)
    
    # 确定预测日期列表
    if args.from_date is None and args.dates is None:
        print("错误: 必须指定 --dates 或 --from_date")
        sys.exit(1)
    elif args.from_date is not None and args.dates is not None:
        print("警告: 同时指定了 --from_date 和 --dates，将优先使用 --from_date")
        today = int(datetime.now().strftime('%Y%m%d'))
        si = StockInfo(TOKEN)
        trade_dates_df = si.get_trade_open_dates(int(args.from_date), today)
        args.dates = trade_dates_df['trade_date'].astype(int).tolist()
        print(f"预测从 {args.from_date} 到今天的所有交易日，共 {len(args.dates)} 天")
    elif args.from_date is not None:
        today = int(datetime.now().strftime('%Y%m%d'))
        si = StockInfo(TOKEN)
        trade_dates_df = si.get_trade_open_dates(int(args.from_date), today)
        args.dates = trade_dates_df['trade_date'].astype(int).tolist()
        print(f"预测从 {args.from_date} 到今天的所有交易日，共 {len(args.dates)} 天")
    else:
        print(f"预测指定的日期: {args.dates}")
    
    # 初始化 StockInfo
    si = StockInfo(TOKEN)
    
    print("\n正在加载数据集...")
    # 创建门控数据集 (BINARY_T1L10)
    ds_gate = StockDataset(
        idx_code_list=IDX_CODE_LIST,
        rel_code_list=[],
        si=si,
        train_size=1,
        if_update_scaler=False,
        ts_code=args.stock_code,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_type=gate_feature_type,
        predict_type=gate_predict_type
    )
    
    # 创建回归数据集 (REGRESS_T1L)
    ds_reg = StockDataset(
        idx_code_list=IDX_CODE_LIST,
        rel_code_list=[],
        si=si,
        train_size=1,
        if_update_scaler=False,
        ts_code=args.stock_code,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_type=reg_feature_type,
        predict_type=reg_predict_type
    )
    
    print("\n正在加载模型...")
    # 加载模型
    try:
        m_gate = load_model_by_params(args.stock_code, model_type, gate_predict_type, gate_feature_type)
        print(f"✓ 门控模型加载成功: {gate_predict_type}")
    except Exception as e:
        print(f"✗ 门控模型加载失败: {e}")
        print(f"请确保已训练 {args.stock_code} 的 {gate_predict_type} 模型")
        sys.exit(1)
    
    try:
        m_reg = load_model_by_params(args.stock_code, model_type, reg_predict_type, reg_feature_type)
        print(f"✓ 回归模型加载成功: {reg_predict_type}")
    except Exception as e:
        print(f"✗ 回归模型加载失败: {e}")
        print(f"请确保已训练 {args.stock_code} 的 {reg_predict_type} 模型")
        sys.exit(1)
    
    print(f"\n开始软门控预测 (y_base={args.y_base}, gamma={args.gamma})...")
    print("="*80)
    
    # 执行软门控预测
    try:
        accuracy, _, _, mae, std = print_predict_result_soft_gated_t1l10(
            t_list=args.dates,
            ds_gate=ds_gate,
            m_gate=m_gate,
            ds_reg=ds_reg,
            m_reg=m_reg,
            y_base=args.y_base,
            gamma=args.gamma
        )
        
        print("\n" + "="*80)
        print("软门控预测完成！")
        print(f"总体统计:")
        print(f"  - 正确率: {accuracy:.2%}")
        if mae is not None:
            print(f"  - 平均残差(MAE): {mae:.2f} 百分点")
        if std is not None:
            print(f"  - 预测值标准差: {std:.2f} 百分点")
        print("="*80)
        
    except Exception as e:
        print(f"\n软门控预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用 TensorFlow 警告
    main()
