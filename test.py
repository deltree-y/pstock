from utils.const_def import ALL_CODE_LIST
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.residual_lstm import ResidualLSTMModel
from utils.tk import TOKEN
from utils.utils import FeatureType, PredictType, setup_logging

setup_logging()
si = StockInfo(TOKEN)

# 假设你要做T1低于-1%的二分类
ds = StockDataset(
    ts_code='600036.SH',
    idx_code_list=['000001.SH'],
    rel_code_list=ALL_CODE_LIST,
    si=si,
    start_date='20180104',
    end_date='20250530',
    train_size=1,  # 全量数据用于训练，极限过拟合
    feature_type=FeatureType.T1L10_F55,
    predict_type=PredictType.BINARY_T1_L10
)
x, y = ds.normalized_windowed_train_x, ds.train_y

# 用大模型极限训练
model = ResidualLSTMModel(
    x=x, y=y,
    test_x=x, test_y=y,  # 训练集做验证集
    depth=6,             # 层数大
    base_units=128,      # 每层宽
    dropout_rate=0.01,   # 减少正则
    l2_reg=1e-8,         # 取消L2正则
    loss_type='binary_crossentropy',
    class_weights=None,  # 不加权
    predict_type=PredictType.BINARY_T1_L10
)
model.train(x, y, epochs=200, batch_size=1024, learning_rate=0.01, patience=20)

# 打印训练集准确率
print("训练集最终准确率：", model.history.val_t1_accu[-1])