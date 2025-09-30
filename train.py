# coding=utf-8
import os
import random
import numpy as np
import tensorflow as tf
import torch
from sklearn.utils import compute_class_weight
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.analyze import plot_l2_loss_curves, print_recall_score
from model.residual_lstm import ResidualLSTMModel
from model.residual_tcn import ResidualTCNModel
from model.transformer import TransformerModel
from predicproc.show import print_predict_result
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, BASE_DIR, MODEL_DIR, NUM_CLASSES
from utils.utils import FeatureType, PredictType, setup_logging, print_ratio

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def train_and_record_l2(model_type, l2_reg, tx, ty, vx, vy, model_params, train_params):
    if model_type == 'residual_lstm':
        model = ResidualLSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    elif model_type == 'residual_tcn':
        model = ResidualTCNModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    elif model_type == 'transformer':
        model = TransformerModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    print(f"[INFO] Start training: l2_reg={l2_reg}")
    model.train(tx=tx, ty=ty, **train_params)
    val_losses = model.history.val_losses
    return val_losses, model

def auto_search():
    setup_logging()
    set_seed(42)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # ===== 数据准备 =====
    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = ['000001.SH']
    related_stock_list = ALL_CODE_LIST
    predict_type = PredictType.BINARY_T1_L10
    t_list = (si.get_trade_open_dates('20250801', '20250829'))['trade_date'].tolist()

    ds = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=related_stock_list, si=si,
                      start_date='20190104', end_date='20250903',
                      train_size=0.9,
                      feature_type=FeatureType.EXTRA_55,
                      predict_type=predict_type)
    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    ty, vy = ty[:, 0], vy[:, 0]

    # ===== 模型参数 =====
    model_type = 'residual_lstm'  # 可选: 'residual_lstm', 'residual_tcn', 'transformer', 'mini'
    p = 2
    dropout_rate = 0.3
    # 残差LSTM模型参数
    depth = 6#dp#6
    base_units = 32#bu#32
    # Transformer模型参数
    d_model = 256
    num_heads = 4
    ff_dim = 512
    num_layers = 4#nl#4
    # TCN模型参数
    nb_stacks = 2#ns  #增大nb_stacks会整体重复残差结构，直接增加模型深度
    dilations = [1, 2, 4, 8, 16]#, 32]    #[1, 2, 4, 8] #每个stack内的dilation设置，增大dilation可以让模型看到更长时间的历史
    nb_filters = 64 #有多少组专家分别提取不同类型的特征
    kernel_size = 8 #每个专家一次能看到多长时间的历史窗口
    
    # ===== 训练参数 =====
    epochs = 120
    batch_size = 2048
    patience = 30
    learning_rate = 0.001
    loss_type = 'binary_crossentropy'

    # ===== 根据上面的选择和参数自动配置模型参数 =====
    if predict_type.is_classify():
        class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=ty)
        cls_weights = dict(enumerate(class_weights))
    else:
        cls_weights = None

    if model_type == 'residual_lstm':
        model_params = dict(
            p=p, dropout_rate=dropout_rate, depth=depth, base_units=base_units,
            use_se=True, se_ratio=8, class_weights=cls_weights, loss_type=loss_type, predict_type=predict_type
        )
    elif model_type == 'residual_tcn':
        model_params = dict(
            p=p, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
            dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, predict_type=predict_type
        )
    elif model_type == 'transformer':
        model_params = dict(
            d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, num_layers=num_layers,
            use_gating=True, use_pos_encoding=True, class_weights=cls_weights, loss_type=loss_type, predict_type=predict_type
        )
    else:
        raise ValueError("unknown model_type")
    train_params = dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)

    # ===== 搜索 =====
    l2_reg_list = [0.00007]
    feature_type_list = [FeatureType.ALL, FeatureType.EXTRA_16, FeatureType.EXTRA_19, FeatureType.EXTRA_24, FeatureType.EXTRA_32, FeatureType.EXTRA_37,\
                         FeatureType.EXTRA_15, FeatureType.EXTRA_20, FeatureType.EXTRA_25, FeatureType.EXTRA_30, FeatureType.EXTRA_35, FeatureType.EXTRA_45, FeatureType.EXTRA_55]
    history_dict = {}
    best_feature, best_val, best_model = None, float('inf'), None

    for ft in feature_type_list:
        for l2_reg in l2_reg_list:
            ds = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=related_stock_list, si=si,
                            start_date='20190104', end_date='20250903',
                            train_size=0.9,
                            feature_type=ft,
                            predict_type=predict_type)
            tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
            ty, vy = ty[:, 0], vy[:, 0]

            print(f"\n{'='*5} 开始训练: model={model_type} {'='*5}")
            print(f"{'='*5} 训练参数: batch={batch_size}, lr={learning_rate}, drop={dropout_rate}, l2={l2_reg}, dep={depth},  bu={base_units}, patience={patience} {'='*5}")
            print(f"{'='*5} 模型参数: feature={ft} {model_params} {'='*5}\n")
            save_path = os.path.join(BASE_DIR, MODEL_DIR, f"{primary_stock_code}_{model_type}_{predict_type}_ft{ft}.h5")

            val_losses, model = train_and_record_l2(model_type, l2_reg, tx, ty, vx, vy, model_params, train_params)
            history_dict[l2_reg] = {'val_loss': val_losses}
            min_val = np.min(val_losses)
            print(f"[INFO] feature={ft}, min val_loss={min_val:.4f}")
            if min_val < best_val:
                best_feature, best_val, best_model = ft, min_val, model
            print_predict_result(t_list, ds, model, predict_type)
            vx_pred_raw = model.model.predict(vx)
            print_recall_score(vx_pred_raw, vy, predict_type)


    print(f"\n[RESULT] Best : {best_feature}, min val_loss: {best_val:.4f}")
    plot_l2_loss_curves(history_dict, epochs)
    # 可选: 保存最佳模型
    best_model.save(save_path)

if __name__ == "__main__":
    auto_search()