# PStock - AI股票预测系统

一个基于深度学习的股票价格预测系统，使用LSTM、TCN等神经网络模型进行股票走势预测。

## 功能特点

- 📈 多模型支持：LSTM、Residual LSTM、TCN、Transformer等
- 📊 多维度数据：股票价格、技术指标、宏观经济数据
- 🎯 分类预测：将价格变化分为多个等级进行预测
- 📉 风险控制：基于历史数据的风险评估
- 🔄 实时更新：支持从TuShare获取最新数据

## 项目结构

```
pstock/
├── datasets/           # 数据处理模块
│   ├── stock.py       # 股票数据类
│   ├── dataset.py     # 数据集处理
│   ├── stockinfo.py   # 股票信息获取
│   └── ...
├── model/             # 模型定义
│   ├── residual_lstm.py  # 残差LSTM模型
│   ├── tcn.py           # 时序卷积网络
│   └── ...
├── predicproc/        # 预测处理
│   ├── predict.py     # 预测逻辑
│   └── analyze.py     # 结果分析
├── utils/             # 工具函数
│   ├── const_def.py   # 常量定义
│   ├── utils.py       # 通用工具
│   └── tk.py          # API配置 (需要配置)
├── new_train.py       # 训练脚本
├── predict.py         # 预测脚本
└── requirements.txt   # 依赖包
```

## 安装与配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API令牌

1. 注册 [TuShare Pro](https://tushare.pro/) 账号
2. 获取API令牌
3. 复制模板文件并配置令牌：

```bash
cp utils/tk.py.template utils/tk.py
```

4. 编辑 `utils/tk.py` 文件，将 `YOUR_TUSHARE_TOKEN_HERE` 替换为您的实际令牌

```python
# utils/tk.py
TOKEN = "your_actual_tushare_token_here"
```

**注意**: `utils/tk.py` 文件已在 `.gitignore` 中排除，不会被提交到版本控制。

### 3. 创建数据目录

```bash
mkdir -p data/{stock,index,model,scaler,bins,global,temp}
```

## 使用方法

### 训练模型

```bash
python new_train.py
```

### 预测股票

```bash
python predict.py
```

## 模型说明

### 支持的模型
- **Residual LSTM**: 残差连接的LSTM网络，解决梯度消失问题
- **TCN**: 时序卷积网络，适合长序列建模
- **Transformer**: 基于注意力机制的模型

### 数据特征
- 价格数据：开盘价、收盘价、最高价、最低价、成交量
- 技术指标：移动平均线、RSI、MACD等
- 宏观数据：利率、汇率、宏观经济指标

### 预测目标
- T1低值变化率：下一个交易日的最低价变化
- T2高值变化率：下两个交易日的最高价变化

## 配置说明

主要配置文件：`utils/const_def.py`

```python
NUM_CLASSES = 5          # 分类数量
CONTINUOUS_DAYS = 50     # 时间窗口大小
BASE_DIR = 'data'        # 数据根目录
```

## 注意事项

1. **API限制**: TuShare API有调用频率限制，请合理使用
2. **数据质量**: 确保数据的完整性和准确性
3. **风险提示**: 预测结果仅供参考，投资有风险
4. **环境要求**: 推荐使用Python 3.8+

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

---

**免责声明**: 本系统提供的预测结果仅供参考，不构成投资建议。股市有风险，投资需谨慎。