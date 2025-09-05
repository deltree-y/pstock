#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PStock 使用示例

这个脚本展示了如何使用PStock系统进行股票预测的基本流程。
在运行之前，请确保已经配置好Tushare API Token。
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "datasets"))
sys.path.append(str(project_root / "utils"))

from datasets.stockinfo import StockInfo
from datasets.dataset import StockDataset
from model.lstmmodel import LSTMModel
from predicproc.predict import Predict
from utils.const_def import TOKEN
from utils.utils import setup_logging

def example_basic_usage():
    """
    基本使用示例：训练模型并进行预测
    """
    print("🚀 PStock 基本使用示例")
    print("=" * 50)
    
    # 设置日志
    setup_logging()
    
    # 配置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    try:
        # 1. 初始化股票信息服务
        print("📊 初始化股票信息服务...")
        si = StockInfo(TOKEN)
        
        # 2. 选择要预测的股票（中国银行）
        stock_code = '600036.SH'
        print(f"📈 选择股票: {stock_code}")
        
        # 3. 创建数据集
        print("🔧 创建数据集...")
        ds = StockDataset(
            stock_code, 
            si, 
            start_date='20200101',
            end_date='20241201', 
            train_size=0.9
        )
        
        # 4. 获取训练和测试数据
        print("📚 准备训练数据...")
        tx, ty = ds.normalized_windowed_train_x, ds.train_y
        vx, vy = ds.normalized_windowed_test_x, ds.test_y
        
        print(f"   训练集形状: X={tx.shape}, Y={ty.shape}")
        print(f"   测试集形状: X={vx.shape}, Y={vy.shape}")
        
        # 5. 创建LSTM模型
        print("🤖 创建LSTM模型...")
        model = LSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, p=2)
        
        # 6. 训练模型（示例用小的epoch数）
        print("🏃‍♂️ 开始训练模型...")
        train_result = model.train(epochs=10, batch_size=32)
        print(f"   训练结果: {train_result}")
        
        # 7. 获取训练指标
        best_val_acc = model.history.get_best_val()
        last_loss, last_val_loss = model.history.get_last_loss()
        best_val_loss = model.history.get_best_val_loss()
        
        print(f"📊 训练指标:")
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   最终训练损失: {last_loss:.4f}")
        print(f"   最终验证损失: {last_val_loss:.4f}")
        print(f"   最佳验证损失: {best_val_loss:.4f}")
        
        # 8. 进行预测
        print("🔮 进行预测...")
        predict_dates = ['20241125', '20241126', '20241127']
        
        for date in predict_dates:
            try:
                print(f"\n📅 预测日期: {date}")
                data, base_price = ds.get_predictable_dataset_by_date(date)
                pred_data = model.model(data)
                
                # 创建预测处理器
                predictor = Predict(
                    pred_data, 
                    base_price, 
                    ds.bins1.prop_bins, 
                    ds.bins2.prop_bins
                )
                
                # 显示预测结果
                predictor.print_predict_result()
                
            except Exception as e:
                print(f"   预测失败: {e}")
        
        # 9. 保存模型（可选）
        model_path = project_root / "data" / "model" / f"{stock_code}_example.h5"
        os.makedirs(model_path.parent, exist_ok=True)
        model.save(str(model_path))
        print(f"💾 模型已保存至: {model_path}")
        
        print("\n✅ 示例运行完成！")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("请检查:")
        print("1. Tushare API Token是否正确配置")
        print("2. 网络连接是否正常")
        print("3. 依赖包是否已安装")

def example_data_download():
    """
    数据下载示例
    """
    print("📥 数据下载示例")
    print("=" * 30)
    
    try:
        from datasets.stock import Stocks
        
        # 初始化股票信息
        si = StockInfo(TOKEN)
        
        # 获取市值过滤后的股票列表
        stock_list = si.get_filtered_stock_list(mmv=5000000)  # 50亿市值以上
        print(f"📋 筛选出股票数量: {len(stock_list)}")
        print(f"部分股票代码: {stock_list[:5]}")
        
        # 下载最近30天的数据
        from datetime import datetime, timedelta
        end_date = datetime.today().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=30)).strftime('%Y%m%d')
        
        print(f"📅 下载数据时间段: {start_date} ~ {end_date}")
        
        # 选择前5只股票进行下载示例
        sample_stocks = stock_list[:5]
        stocks = Stocks(sample_stocks, si, start_date=start_date, end_date=end_date)
        
        print("✅ 数据下载完成！")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")

def example_model_prediction():
    """
    使用已训练模型进行预测的示例
    """
    print("🔮 模型预测示例")
    print("=" * 30)
    
    try:
        # 加载已训练的模型
        model_path = project_root / "data" / "model" / "600036.SH_example.h5"
        
        if not model_path.exists():
            print("⚠️  示例模型不存在，请先运行基本使用示例进行训练")
            return
        
        # 加载模型
        model = LSTMModel(fn=str(model_path))
        print(f"📂 模型加载成功: {model_path}")
        
        # 准备预测数据
        si = StockInfo(TOKEN)
        ds = StockDataset('600036.SH', si, start_date='20240101', end_date='20241201')
        
        # 进行预测
        date = '20241127'
        data, base_price = ds.get_predictable_dataset_by_date(date)
        pred_data = model.model(data)
        
        # 处理预测结果
        predictor = Predict(pred_data, base_price, ds.bins1.prop_bins, ds.bins2.prop_bins)
        predictor.print_predict_result()
        
        print("✅ 预测完成！")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")

if __name__ == "__main__":
    print("🏦 PStock - 基于LSTM的股票预测系统")
    print("=" * 60)
    
    # 检查Token配置
    if not TOKEN or TOKEN == 'your_tushare_token_here':
        print("⚠️  请先在 utils/const_def.py 中配置您的Tushare API Token")
        print("   获取Token: https://tushare.pro/register")
        sys.exit(1)
    
    print("请选择要运行的示例:")
    print("1. 基本使用示例（数据准备 -> 训练 -> 预测）")
    print("2. 数据下载示例")
    print("3. 模型预测示例")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (0-3): ").strip()
            
            if choice == '0':
                print("👋 再见！")
                break
            elif choice == '1':
                example_basic_usage()
            elif choice == '2':
                example_data_download()
            elif choice == '3':
                example_model_prediction()
            else:
                print("❌ 无效选择，请输入 0-3")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 运行错误: {e}")