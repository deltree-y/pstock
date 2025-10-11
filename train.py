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
from model.utils import get_model_file_name
from predicproc.show import print_predict_result
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, NUM_CLASSES, IDX_CODE_LIST
from utils.utils import FeatureType, PredictType, ModelType, setup_logging, print_ratio

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def train_and_record_l2(model_type, l2_reg, tx, ty, vx, vy, model_params, train_params):
    if model_type == ModelType.RESIDUAL_LSTM:
        model = ResidualLSTMModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    elif model_type == ModelType.RESIDUAL_TCN:
        model = ResidualTCNModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    elif model_type == ModelType.TRANSFORMER:
        model = TransformerModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    print(f"[INFO] Start training:")# l2_reg={l2_reg}")
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
    index_code_list = IDX_CODE_LIST
    related_stock_list = ALL_CODE_LIST
    t_list = (si.get_trade_open_dates('20250101', '20250920'))['trade_date'].tolist()
    t_start_date, t_end_date = '20180104', '20250530'

    # ===== 模型参数 =====
    model_type = ModelType.RESIDUAL_LSTM  # 可选: 'residual_lstm', 'residual_tcn', 'transformer', 'mini'
    p = 2
    dropout_rate = 0.3
    # 残差LSTM模型参数
    depth = 6#dp#6
    base_units = 64#bu#32
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
    epochs = 100
    batch_size = 1024
    patience = 20
    learning_rate = 0.0002
    train_size = 0.9
    loss_type = 'binary_crossentropy' #focal_loss,binary_crossentropy


    # ===== 搜索 =====
    l2_reg_list = [0.00007]
    lr_list = [0.0002]#, 0.0001, 0.0005, 0.001, 0.005]
    feature_type_list = [FeatureType.T1L10_F55]
    predict_type_list = [PredictType.BINARY_T1_L10]
    history_dict = {}
    best_paras, best_val, best_model = None, float('inf'), None
    
    for depth,base_units in zip([2, 4], [128, 128]):  # depth, base_units
        for lr in lr_list:
            for pt in predict_type_list:
                for ft in feature_type_list:
                    for l2_reg in l2_reg_list:
                        paras = f"{pt}_{ft}_{l2_reg}_{lr}_{depth}_{base_units}"
                        print(f"\n{'='*5} 开始训练处理: model={model_type} {'='*5}")

                        ds = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=related_stock_list, si=si,start_date=t_start_date, end_date=t_end_date,train_size=train_size,feature_type=ft,predict_type=pt)
                        ds_pred = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=[], si=si, if_update_scaler=False, start_date='19930101', end_date='20251010', train_size=1, feature_type=ft, predict_type=pt)
                        t_list = [d for d in t_list if np.where(ds_pred.raw_data[:, 0] == d)[0][0] + ds_pred.window_size <= ds_pred.raw_data.shape[0]]                        
                        tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
                        ty, vy = ty[:, 0], vy[:, 0]

                        # ===== 根据上面的选择和参数自动配置模型参数 =====
                        if pt.is_classify():
                            class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=ty)
                            cls_weights = dict(enumerate(class_weights))
                        else:
                            cls_weights = None
                            #cls_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=ty)
                            #cls_weights[0] *= 2  # 加大类别0权重

                        if model_type == ModelType.RESIDUAL_LSTM:
                            model_params = dict(
                                p=p, dropout_rate=dropout_rate, depth=depth, base_units=base_units,
                                use_se=True, se_ratio=8, class_weights=cls_weights, loss_type=loss_type, predict_type=pt
                            )
                        elif model_type == ModelType.RESIDUAL_TCN:
                            model_params = dict(
                                p=p, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations,
                                dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, predict_type=pt
                            )
                        elif model_type == ModelType.TRANSFORMER:
                            model_params = dict(
                                d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, num_layers=num_layers,
                                use_gating=True, use_pos_encoding=True, class_weights=cls_weights, loss_type=loss_type, predict_type=pt
                            )
                        else:
                            raise ValueError("unknown model_type")
                        train_params = dict(epochs=epochs, batch_size=batch_size, learning_rate=lr, patience=patience)
                        save_path = get_model_file_name(primary_stock_code, model_type, pt, ft)
                        print(f"{'='*5} 训练参数: batch={batch_size}, lr={lr}, drop={dropout_rate}, l2={l2_reg}, dep={depth},  bu={base_units}, patience={patience} {'='*5}")
                        print(f"{'='*5} 模型参数: feature={ft}, model_params={model_params} {'='*5}\n")
                        print_ratio(ty, "验证集(ty)")
                        
                        #开始训练
                        val_losses, model = train_and_record_l2(model_type, l2_reg, tx, ty, vx, vy, model_params, train_params)
                        history_dict[paras] = {'val_loss': val_losses}
                        min_val = np.min(val_losses)
                        print(f"[INFO] paras={paras}, min val_loss={min_val:.4f}")
                        if min_val < best_val:
                            best_paras, best_val, best_model = paras, min_val, model
                        print_predict_result(t_list, ds_pred, model, pt)
                        vx_pred_raw = model.model.predict(vx)
                        print_recall_score(vx_pred_raw, vy, pt)

    print(f"\n[RESULT] Best : {best_paras}, min val_loss: {best_val:.4f}")
    best_model.save(save_path)
    plot_l2_loss_curves(history_dict, epochs)

if __name__ == "__main__":
    auto_search()