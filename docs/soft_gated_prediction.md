# 软门控预测功能文档

## 概述

软门控预测是一个二阶段预测方法，它结合了二分类模型（BINARY_T1L10）和回归模型（REGRESS_T1L）的优势，通过使用二分类模型的概率输出来加权调整回归模型的预测结果。

## 原理

### 融合公式

```
p_g = p ** gamma
y_hat = p_g * y_reg + (1 - p_g) * y_base
```

其中：
- `p`: 二分类模型（BINARY_T1L10）输出的概率 [0, 1]
- `gamma`: 锐化参数，默认 1.0（不锐化）
- `p_g`: 门控权重
- `y_reg`: 回归模型（REGRESS_T1L）输出的百分点变化率
- `y_base`: 基础回归值（百分点），默认 -0.2%
- `y_hat`: 融合后的预测值（百分点）

### 工作机制

1. 当二分类模型认为会跌破阈值（p 接近 1）时，更多地信任回归模型的预测
2. 当二分类模型不确定或认为不会跌破（p 接近 0）时，预测值会向 y_base 偏移
3. gamma 参数可以调整这个门控的"锐度"：
   - gamma = 1.0: 线性门控（默认）
   - gamma > 1.0: 更锐化，增强高置信度预测的权重
   - gamma < 1.0: 更平滑，减弱置信度的影响

## 使用方法

### 基本用法

```python
from datasets.stockinfo import StockInfo
from datasets.dataset import StockDataset
from model.utils import load_model_by_params
from utils.utils import FeatureType, ModelType, PredictType
from predicproc.show import print_predict_result_soft_gated_t1l10

# 初始化
si = StockInfo(TOKEN)
stock_code = "600036.SH"
dates = [20230601, 20230602, 20230605]

# 创建数据集
ds_gate = StockDataset(
    ts_code=stock_code,
    idx_code_list=IDX_CODE_LIST,
    rel_code_list=[],
    si=si,
    train_size=1,
    if_update_scaler=False,
    feature_type=FeatureType.BINARY_T1L10_F55,
    predict_type=PredictType.BINARY_T1L10
)

ds_reg = StockDataset(
    ts_code=stock_code,
    idx_code_list=IDX_CODE_LIST,
    rel_code_list=[],
    si=si,
    train_size=1,
    if_update_scaler=False,
    feature_type=FeatureType.REGRESS_T1L_F55,
    predict_type=PredictType.REGRESS_T1L
)

# 加载模型
m_gate = load_model_by_params(stock_code, ModelType.TRANSFORMER, 
                               PredictType.BINARY_T1L10, 
                               FeatureType.BINARY_T1L10_F55)
m_reg = load_model_by_params(stock_code, ModelType.TRANSFORMER,
                              PredictType.REGRESS_T1L,
                              FeatureType.REGRESS_T1L_F55)

# 执行软门控预测
accuracy, _, _, mae, std = print_predict_result_soft_gated_t1l10(
    t_list=dates,
    ds_gate=ds_gate,
    m_gate=m_gate,
    ds_reg=ds_reg,
    m_reg=m_reg,
    y_base=-0.2,  # 默认值
    gamma=1.0     # 默认值
)

print(f"正确率: {accuracy:.2%}")
print(f"平均残差: {mae:.2f} 百分点")
print(f"标准差: {std:.2f} 百分点")
```

### 使用测试脚本

项目提供了一个便捷的测试脚本 `test_soft_gated.py`：

```bash
# 预测特定日期
python test_soft_gated.py --stock_code 600036.SH --dates 20230601 20230602 20230605

# 预测从指定日期到今天的所有交易日
python test_soft_gated.py --stock_code 600036.SH --from_date 20230601

# 自定义参数
python test_soft_gated.py \
    --stock_code 600036.SH \
    --dates 20230601 20230602 \
    --y_base -0.3 \
    --gamma 1.2 \
    --model_type TRANSFORMER
```

### 参数说明

#### 必需参数
- `t_list`: 要预测的日期列表（格式：YYYYMMDD）
- `ds_gate`: 二分类数据集（BINARY_T1L10）
- `m_gate`: 二分类模型（BINARY_T1L10）
- `ds_reg`: 回归数据集（REGRESS_T1L）
- `m_reg`: 回归模型（REGRESS_T1L）

#### 可选参数
- `y_base`: 基础回归值（百分点），默认 -0.2
  - 当门控概率低时，预测会向此值偏移
  - 建议设置为历史平均跌幅或保守估计值
- `gamma`: 锐化参数，默认 1.0
  - 控制门控的锐度
  - 通常保持在 [0.5, 2.0] 范围内

## 输出说明

### 实时输出
函数执行时会实时输出每个日期的预测结果：
- `o`: 预测正确（残差 ≤ 0.2 百分点）
- `x`: 预测错误（残差 > 0.2 百分点）

### 统计结果
执行完成后输出：
1. **预测错误列表**: 列出所有预测错误的日期和详细信息
2. **正确率**: 预测正确的比例
3. **平均残差(MAE)**: 预测值与真实值的平均绝对误差（百分点）
4. **预测值标准差**: 预测值的离散程度（百分点）

### 返回值
```python
(accuracy, None, None, mae, std)
```
- `accuracy`: 正确率 (float)
- `None`: 占位符（保持与 `print_predict_result` 接口一致）
- `None`: 占位符（保持与 `print_predict_result` 接口一致）
- `mae`: 平均残差（百分点）
- `std`: 预测值标准差（百分点）

## 应用场景

### 适用情况
1. 已经训练好 BINARY_T1L10 和 REGRESS_T1L 两个模型
2. 需要结合分类置信度来调整回归预测
3. 希望在不确定时采用保守策略（y_base）

### 优势
1. **降低风险**: 在分类模型不确定时，采用保守的基准值
2. **提高准确性**: 在分类模型高置信度时，充分利用回归模型的精确预测
3. **灵活性**: 可通过 gamma 参数调整门控策略

### 调参建议
1. **y_base**: 根据股票历史表现设置
   - 稳定股票: -0.1 到 -0.2
   - 波动股票: -0.3 到 -0.5
2. **gamma**: 根据模型可靠性调整
   - 模型可靠: 1.2 - 1.5（更锐化）
   - 模型不稳定: 0.8 - 1.0（更平滑）

## 注意事项

1. **数据一致性**: 确保两个数据集使用相同的日期范围和数据源
2. **模型要求**: 必须预先训练好对应的模型文件
3. **基准价验证**: 函数会检查两个数据集的基准价是否一致，如有差异会发出警告
4. **单位统一**: 所有百分点单位保持一致（如 -0.5 表示 -0.5%）

## 故障排查

### 模型加载失败
```
错误: 模型加载失败
解决: 确认已训练并保存对应股票的模型文件
```

### 日期数据不可用
```
错误: 日期 XXXXXXXX 数据不可用
解决: 检查数据集的日期范围，确保目标日期在范围内且有足够的历史窗口
```

### 基准价不一致警告
```
警告: 日期XXXXXXXX: 门控bp=XX.XX 与回归bp=XX.XX 不一致
说明: 两个数据集的基准价格有差异，可能影响预测准确性
解决: 检查数据集初始化参数是否一致
```

## 示例输出

```
--------------------------------------------------------------------------------

软门控预测结果如下 (y_base=-0.2, gamma=1.0):
ooxoxoooxxo...

预测错误列表:
T0[20230602] - [真实/预测值]__跌幅(差值) : [11.23/11.45]  -0.45%/-0.20%(-0.25)
T0[20230605] - [真实/预测值]__跌幅(差值) : [10.98/11.32]  -0.65%/-0.35%(-0.30)
...

正确率: 75.00%, 正确个数: 9/12
平均残差(MAE): 0.15 百分点
预测值标准差: 0.42 百分点
----------------------------------------------------------------------------------------------------
```

## 扩展开发

如需定制化开发，可以参考 `print_predict_result_soft_gated_t1l10` 的实现：
1. 修改融合公式以适应不同场景
2. 添加更多统计指标
3. 支持其他模型组合（如 BINARY_T1H10 + REGRESS_T1H）
4. 集成到自动化交易系统

## 参考

- `predicproc/show.py`: 主函数实现
- `predicproc/predict.py`: Predict 类实现
- `test_soft_gated.py`: 完整使用示例
