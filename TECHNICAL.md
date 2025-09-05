# PStock 技术文档

## 详细技术说明

### 数据流程

1. **数据获取**: 从Tushare API获取股票数据
2. **特征工程**: 计算技术指标和特征
3. **数据预处理**: 归一化和窗口化
4. **模型训练**: 使用LSTM进行训练
5. **预测输出**: 生成分类预测结果

### 关键技术特性

#### LSTM模型架构
```python
# 双向LSTM结构
- 第一层: Bidirectional LSTM (p*32 units)
- 第二层: Bidirectional LSTM (p*16 units) 
- 正则化: L2正则化 + LayerNormalization + Dropout(0.5)
- 输出层: 两个Dense层分别预测T1L和T2H
```

#### 预测分类系统
- **类别数量**: 20个类别 (NUM_CLASSES = 20)
- **预测目标**: 
  - T1L: 短期最低价格变动预测
  - T2H: 中期最高价格变动预测
- **输出格式**: 每个类别的概率分布

#### 数据特征
- **时间窗口**: 25个交易日 (CONTINUOUS_DAYS = 25)
- **输入特征**: 包括价格、成交量、技术指标等
- **数据分割**: 默认80%训练，20%测试

### 使用示例

#### 基本使用流程

```python
# 1. 初始化股票信息
from datasets.stockinfo import StockInfo
from utils.const_def import TOKEN
si = StockInfo(TOKEN)

# 2. 创建数据集
from dataset import StockDataset
primary_stock_code = '600036.SH'
ds = StockDataset(primary_stock_code, si, 
                 start_date='20200101', 
                 end_date='20241201', 
                 train_size=0.9)

# 3. 获取训练数据
tx, ty = ds.normalized_windowed_train_x, ds.train_y
vx, vy = ds.normalized_windowed_test_x, ds.test_y

# 4. 创建和训练模型
from model.lstmmodel import LSTMModel
tm = LSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, p=4)
tm.train(epochs=100, batch_size=32)

# 5. 保存模型
tm.save("model_path.h5")

# 6. 进行预测
data, bp = ds.get_predictable_dataset_by_date('20241201')
pred_data = tm.model(data)
```

#### 预测结果解释

```python
from predicproc.predict import Predict
# 创建预测处理器
predictor = Predict(pred_data, base_price, bins1, bins2)
# 打印预测结果
predictor.print_predict_result()
```

输出示例:
```
Predict y1l label[15] pct min/avg/max is <2.15%/2.87%/3.42%> 
price is <23.45/23.78/24.12>
```

### 配置参数详解

#### 股票代码配置
```python
# 银行股票代码列表
BANK_CODE_LIST = ['000001.SZ', '601288.SH', ...]

# 相关股票代码
REL_CODE_LIST = BANK_CODE_LIST + STOCK_CODE_LIST + ...
```

#### 模型参数
```python
# 训练参数
epochs = 100          # 训练轮数
batch_size = 32       # 批次大小  
p = 4                 # 网络复杂度参数

# 数据参数
window_size = 25      # 时间窗口大小
train_size = 0.8      # 训练集比例
```

#### 预测阈值
```python
T1_LOWER, T1_UPPER = 0.0, 0.005    # T1预测阈值
T2_LOWER, T2_UPPER = 0.015, 0.025  # T2预测阈值
MIN_TOTAL_MV = 3000000              # 最小市值过滤
```

### 文件结构说明

#### 数据存储结构
```
data/
├── stock/          # 个股数据CSV文件
├── index/          # 指数数据CSV文件  
├── model/          # 训练好的模型文件
├── scaler/         # 数据标准化器
├── bins/           # 分箱配置文件
└── temp/           # 临时文件
```

#### 重要类说明

**Stock类** (`datasets/stock.py`)
- 管理单只股票的数据下载和存储
- 支持增量更新和本地缓存
- 自动处理数据格式转换

**Trade类** (`datasets/trade.py`)
- 计算技术指标 (移动平均、RSI等)
- 生成预测目标变量 (T1L, T2H)
- 数据对齐和清洗

**StockDataset类** (`datasets/dataset.py`)
- 构建机器学习数据集
- 数据归一化和窗口化
- 训练/测试集分割

**LSTMModel类** (`model/lstmmodel.py`)
- 双向LSTM网络实现
- 训练过程管理
- 模型保存和加载

### 常见问题

#### Q: 如何添加新的股票代码？
A: 在`utils/const_def.py`中的相应列表中添加股票代码即可。

#### Q: 如何调整预测类别数量？
A: 修改`NUM_CLASSES`常量，并相应调整分箱配置。

#### Q: 预测精度如何评估？
A: 系统会自动计算验证集上的准确率和损失函数值。

#### Q: 如何处理缺失数据？
A: 系统会自动跳过缺失数据的日期，并在日志中记录。

### 性能优化建议

1. **GPU加速**: 如有GPU，设置TensorFlow使用GPU训练
2. **批次大小**: 根据内存大小调整batch_size
3. **数据缓存**: 预处理后的数据会自动缓存
4. **模型复杂度**: 通过参数p调整网络复杂度

### 扩展开发

#### 添加新的技术指标
在`Trade.update_new_feature()`方法中添加新的特征计算。

#### 修改模型结构
在`LSTMModel.create_model()`方法中调整网络结构。

#### 自定义预测目标
修改`Trade`类中的目标变量计算逻辑。