# PStock 配置文件示例
# 复制此文件为 config.py 并根据需要修改

# Tushare API配置
TUSHARE_TOKEN = "your_tushare_token_here"

# 模型训练参数
TRAINING_CONFIG = {
    "epochs": 100,           # 训练轮数
    "batch_size": 32,        # 批次大小
    "train_size": 0.9,       # 训练集比例
    "window_size": 25,       # 时间窗口大小
    "model_complexity": 4,   # 模型复杂度参数(p)
}

# 数据配置
DATA_CONFIG = {
    "start_date": "20200101",    # 数据开始日期
    "end_date": "20241201",      # 数据结束日期
    "min_market_value": 5000000, # 最小市值过滤(万元)
}

# 预测配置
PREDICTION_CONFIG = {
    "num_classes": 20,          # 预测类别数
    "prediction_days": 3,       # 预测天数
}

# 股票配置
STOCK_CONFIG = {
    # 主要预测股票
    "primary_stock": "600036.SH",  # 中国银行
    
    # 相关股票列表 (用于特征增强)
    "related_stocks": [
        "000001.SZ",  # 平安银行
        "601288.SH",  # 农业银行  
        "601939.SH",  # 建设银行
        "601988.SH",  # 中国银行
        "600000.SH",  # 浦发银行
    ],
    
    # 行业股票
    "bank_stocks": [
        "000001.SZ", "002142.SZ", "600000.SH", 
        "600015.SH", "600016.SH", "601009.SH",
        "601288.SH", "601328.SH", "601398.SH",
        "601939.SH", "601988.SH", "601998.SH"
    ]
}

# 技术指标配置
TECHNICAL_CONFIG = {
    "moving_averages": [5, 10, 20, 60],  # 移动平均线周期
    "rsi_period": 14,                    # RSI周期
    "macd_config": {
        "fast": 12,
        "slow": 26, 
        "signal": 9
    }
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",                     # 日志级别
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "pstock.log"                 # 日志文件
}

# 存储配置
STORAGE_CONFIG = {
    "data_dir": "data",                  # 数据目录
    "model_dir": "data/model",           # 模型存储目录
    "cache_dir": "data/cache",           # 缓存目录
}

# 风险配置
RISK_CONFIG = {
    "max_position": 0.1,                 # 最大仓位比例
    "stop_loss": -0.05,                  # 止损线
    "take_profit": 0.10,                 # 止盈线
}

# 回测配置
BACKTEST_CONFIG = {
    "initial_capital": 100000,           # 初始资金
    "commission": 0.0003,                # 手续费率
    "start_date": "20230101",            # 回测开始日期
    "end_date": "20241201",              # 回测结束日期
}