# coding=utf-8
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.analyze import plot_l2_loss_curves, print_recall_score
from model.residual_lstm import ResidualLSTMModel
from model.residual_tcn import ResidualTCNModel
from model.transformer import TransformerModel
from model.conv1d import Conv1DResModel
from model.utils import get_model_file_name
from predicproc.show import print_predict_result
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, NUM_CLASSES, IDX_CODE_LIST, BIG_IDX_CODE_LIST, BANK_CODE_LIST_10, ACCU_RATE_THRESHOLD, CODE_LIST_TEMP
from utils.utils import FeatureType, PredictType, ModelType, setup_logging, print_ratio, print_nan_inf_info

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    elif model_type == ModelType.CONV1D:
        model = Conv1DResModel(x=tx, y=ty, test_x=vx, test_y=vy, l2_reg=l2_reg, **model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    print(f"[INFO] Start training:")# l2_reg={l2_reg}")
    model.train(tx=tx, ty=ty, **train_params)
    val_losses = model.history.val_losses
    return val_losses, model

def auto_search():
    setup_logging()
    #set_seed(42)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # ===== 数据准备 =====
    si = StockInfo(TOKEN)
    primary_stock_code = '600036.SH'
    index_code_list = IDX_CODE_LIST#BIG_IDX_CODE_LIST#IDX_CODE_LIST
    related_stock_list = ALL_CODE_LIST#CODE_LIST_TEMP#ALL_CODE_LIST#BANK_CODE_LIST_10#[]#ALL_CODE_LIST
    t_list = (si.get_trade_open_dates('20250101', '20250920'))['trade_date'].astype(str).tolist()
    t_start_date, t_end_date = '20160104', '20250101'

    # ---模型通用参数---
    model_type = ModelType.RESIDUAL_LSTM
    p = 2
    dropout_rate = 0.1
    feature_type_list = [FeatureType.T1L_REG_F50]
    predict_type_list = [PredictType.REGRESS]
    loss_type = 'robust_mse' #focal_loss,binary_crossentropy,mse
    lr_list = [0.0001]#0.0002, 0.0001, 0.0005, 0.001, 0.005]
    l2_reg_list = [0.0001]#[0.00007]
    threshold = 0.5 # 二分类阈值

    # ===== 训练参数 =====
    epochs = 120
    batch_size = 256
    patience = 30
    train_size = 0.9
    cyc = 2    # 搜索轮数
    multiple_cnt = 1    # 数据增强倍数,1表示不增强,4表示增强4倍,最大支持4倍

    # ----- 模型相关参数 ----
    lstm_depth_list, base_units_list = [6], [64]#[6],[64]   # LSTM模型参数 - depth-增大会增加模型深度, base_units-增大每层LSTM单元数
    nb_filters, kernel_size, nb_stacks = [64], [4], [2] # TCN模型参数 - nb_filters-有多少组专家分别提取不同类型的特征, kernel_size-每个专家一次能看到多长时间的历史窗口, nb_stacks-增大会整体重复残差结构，直接增加模型深度, 
    d_model_list, num_heads_list, ff_dim_list, num_layers_list = [32], [4], [128], [2] # Transformer模型参数 - d_model-增大每个时间步的特征维度, num_heads-增大多头注意力机制的头数, ff_dim-增大前馈神经网络的隐藏层维度, num_layers-增大会增加模型深度
    filters_list, kernel_size_list, conv1d_depth_list = [128], [8], [4]   # Conv1D模型参数 - filters-增大每个卷积层的滤波器数量, kernel_size-增大卷积核大小, depth-增大会增加模型深度

    if model_type == ModelType.RESIDUAL_LSTM:# LSTM模型参数 - depth-增大会增加模型深度, base_units-增大每层LSTM单元数
        model_key_params = list(zip(lstm_depth_list, base_units_list, [None]*len(lstm_depth_list), [None]*len(lstm_depth_list)))
    elif model_type == ModelType.RESIDUAL_TCN:# TCN模型参数 - nb_stacks-增大会整体重复残差结构，直接增加模型深度, nb_filters-有多少组专家分别提取不同类型的特征, kernel_size-每个专家一次能看到多长时间的历史窗口
        model_key_params = list(zip(nb_filters, kernel_size, nb_stacks, [None]*len(nb_filters)))
    elif model_type == ModelType.TRANSFORMER:# Transformer模型参数 - d_model-增大每个时间步的特征维度, num_heads-增大多头注意力机制的头数, ff_dim-增大前馈神经网络的隐藏层维度, num_layers-增大会增加模型深度
        model_key_params = list(zip(d_model_list, num_heads_list, ff_dim_list, num_layers_list))
    elif model_type == ModelType.CONV1D:# Conv1D模型参数 - filters-增大每个卷积层的滤波器数量, kernel_size-增大卷积核大小, depth-增大会增加模型深度
        model_key_params = list(zip(filters_list, kernel_size_list, conv1d_depth_list, [None]*len(filters_list)))
    else:
        raise ValueError("unknown model_type")

    # ===== 搜索 =====
    history_dict, history_correction = {}, []
    best_paras, best_val, best_model = None, float('inf'), None
    for cyc_sn in range(cyc):
        print(f"\n ====================================== 搜索轮次: {cyc_sn+1} / {cyc} ======================================")
        for p1,p2,p3,p4 in model_key_params:
            for lr in lr_list:
                for ft, pt in zip(feature_type_list, predict_type_list):
                    for l2_reg in l2_reg_list:
                        # ===== 数据集准备 =====
                        ds = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=related_stock_list, si=si,start_date=t_start_date, end_date=t_end_date,train_size=train_size,feature_type=ft,predict_type=pt)
                        ds_pred = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=[], si=si, if_update_scaler=False, start_date='19930204', end_date='20251010', train_size=1, feature_type=ft, predict_type=pt)
                        t_list = [d for d in t_list if ((idx_arr := np.where(ds_pred.raw_data[:, 0] == d)[0]).size > 0 and idx_arr[0] + ds_pred.window_size <= ds_pred.raw_data.shape[0])]
                        tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
                        ty, vy = ty[:, 0], vy[:, 0]
                        # 数据增强
                        tx, ty = ds.time_series_augmentation_multiple(tx, ty, multiple=multiple_cnt, noise_level=0.01)
                        tx = np.nan_to_num(tx, nan=-1, posinf=-1, neginf=-1)

                        # 检查数据
                        if (np.isnan(tx).sum()+np.isinf(tx).sum()+np.isnan(ty).sum()+np.isinf(ty).sum())>0:
                            raise ValueError("训练集数据包含 NaN 或 Inf, 请检查数据和特征工程")
                        if (np.isnan(vx).sum()+np.isinf(vx).sum()+np.isnan(vy).sum()+np.isinf(vy).sum())>0:
                            raise ValueError("验证集数据包含 NaN 或 Inf, 请检查数据和特征工程")

                        # 检查标签范围
                        print("标签最小:", ty.min(), "标签最大:", ty.max())
                        print(f"DEBUG: ty-{ty[:5]}, \nvy-{vy[:5]}")

                        # ===== 根据上面的选择和参数自动配置模型参数 =====
                        if pt.is_classify():
                            class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=ty)
                            cls_weights = dict(enumerate(class_weights))
                        elif pt.is_binary():
                            classes = np.unique(ty)
                            class_weights = compute_class_weight('balanced', classes=classes, y=ty)
                            cls_weights = {int(c): w for c, w in zip(classes, class_weights)}
                        else:
                            cls_weights = None

                        if model_type == ModelType.RESIDUAL_LSTM:
                            paras = f"{pt}_{ft}_{l2_reg}_{lr}_{p1}_{p2}_{cyc_sn}" 
                            model_params = dict(p=p, depth=p1, base_units=p2, dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, predict_type=pt)
                        elif model_type == ModelType.RESIDUAL_TCN:
                            paras = f"{pt}_{ft}_{l2_reg}_{lr}_{p1}_{p2}_{p3}_{cyc_sn}" 
                            model_params = dict(p=p, nb_filters=p1, kernel_size=p2, nb_stacks=p3, dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, predict_type=pt)
                        elif model_type == ModelType.TRANSFORMER:
                            paras = f"{pt}_{ft}_{l2_reg}_{lr}_{p1}_{p2}_{p3}_{p4}_{cyc_sn}" 
                            model_params = dict(d_model=p1, num_heads=p2, ff_dim=p3, num_layers=p4, dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, predict_type=pt)
                        elif model_type == ModelType.CONV1D:
                            paras = f"{pt}_{ft}_{l2_reg}_{lr}_{p1}_{p2}_{p3}_{cyc_sn}" 
                            model_params = dict(filters=p1, kernel_size=p2, depth=p3, dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, predict_type=pt)
                        else:
                            raise ValueError("unknown model_type")

                        #训练参数配置
                        train_params = dict(epochs=epochs, batch_size=batch_size, learning_rate=lr, patience=patience)
                        save_path = get_model_file_name(primary_stock_code, model_type, pt, ft)#, suffix=cyc_sn)

                        # ===== 训练前数据打印 =====
                        print(f"\n{'='*5} 开始训练处理: model={model_type} {'='*5}")
                        print(f"{'='*5} 训练参数: epochs={epochs}, batch_size={batch_size}, patience={patience}, train_size={train_size} {'='*5}")
                        print(f"{'='*5} 模型参数: feature={ft}, model_params={model_params} {'='*5}\n")
                        if pt.is_classify() or pt.is_binary():
                            print_ratio(ty, "训练集(ty)")
                            print_ratio(vy, "验证集(vy)")
                        
                        #开始训练
                        val_losses, model = train_and_record_l2(model_type, l2_reg, tx, ty, vx, vy, model_params, train_params)
                        history_dict[paras] = {'val_loss': val_losses}
                        min_val = np.min(val_losses)
                        print(f"\n[INFO] paras={paras}, min val_loss={min_val:.4f}")
                        if min_val < best_val:
                            best_paras, best_val, best_model = paras, min_val, model

                        correct_rate, correct_mean_prob, wrong_mean_prob, residual, pred_std = print_predict_result(t_list, ds_pred, model, pt, threshold=threshold)
                        if pt.is_regress():
                            vx_pred_raw = model.model.predict(vx)
                            mae = mean_absolute_error(vy, vx_pred_raw.reshape(-1))
                            r2 = r2_score(vy, vx_pred_raw.reshape(-1))
                            acc_at1pct = np.mean(np.abs(vx_pred_raw.reshape(-1) - vy) <= ACCU_RATE_THRESHOLD)
                            print(f"[回归] 验证MAE: {mae:.4f}, R2: {r2:.4f}, Accuracy%: {acc_at1pct:.2%}")
                            history_correction.append({'para': paras, 'val_loss': min_val, 'mae': mae, 'acc':acc_at1pct, 'correct_rate': correct_rate, 'residual': residual, 'pred_std': pred_std})
                        else:
                            vx_pred_raw = model.model.predict(vx)
                            macro_recall = print_recall_score(vx_pred_raw, vy, pt, threshold=threshold)
                            #best_model.save(f"{save_path}")

                            scores = vx_pred_raw[:, 0]
                            best_thr, best_f1 = 0.5, 0
                            for thr in np.linspace(0.2, 0.8, 600):   # 举例：0.3~0.7 扫一遍
                                pred = (scores > thr).astype(int)
                                f1 = f1_score(vy, pred, average='macro')
                                if f1 > best_f1:
                                    best_f1, best_thr = f1, thr
                            print(f"二分类最优阈值: {best_thr:.3f}, 对应macro F1: {best_f1:.3f}")
                            history_correction.append({'para': paras, 'val_loss': min_val, 'correct_rate': correct_rate, 'correct_mean_prob': correct_mean_prob, 'wrong_mean_prob': wrong_mean_prob, 'macro_recall': macro_recall, 'best_thr': best_thr})
                        
                        for record in history_correction:
                            if pt.is_regress():
                                print(f"[R] p:{model_type}_{record['para']}, vl:{record['val_loss']:.4f}, MAE:{record['mae']:.4f}, Acc v/t:{record['acc']:.2%}/{record['correct_rate']:.2%}, 差均值/标准差:{record['residual']:.2f}/{record['pred_std']:.2f}")
                            else:
                                print(f"[R] p:{model_type}_{record['para']}, vl:{record['val_loss']:.4f}, 最优阈值:{record['best_thr']:.3f}, 正确率/召回率:{record['correct_rate']:.2%}/{record['macro_recall']:.2%}, 正确/错误置信率:{record['correct_mean_prob']:.2f}%/{record['wrong_mean_prob']:.2f}%({record['correct_mean_prob']-record['wrong_mean_prob']:.2f}%)")

    print(f"\n[RESULT] Best : {best_paras}, min val_loss: {best_val:.4f}")
    for record in history_correction:
        if predict_type_list[0].is_regress():
            print(f"[R] p:{model_type}_{record['para']}, vl:{record['val_loss']:.4f}, MAE:{record['mae']:.4f}, Acc t/v:{record['acc']:.2%}/{record['correct_rate']:.2%}, 差均值/标准差:{record['residual']:.2f}/{record['pred_std']:.2f}")
        else:
            print(f"[R] p:{model_type}_{record['para']}, vl:{record['val_loss']:.4f}, 最优阈值:{record['best_thr']:.3f}, 正确率/召回率:{record['correct_rate']:.2%}/{record['macro_recall']:.2%}, 正确/错误置信率:{record['correct_mean_prob']:.2f}%/{record['wrong_mean_prob']:.2f}%({record['correct_mean_prob']-record['wrong_mean_prob']:.2f}%)")
    best_model.save(save_path)
    #plot_l2_loss_curves(history_dict, epochs)
    
if __name__ == "__main__":
    # 允许显存按需增长，避免一次性占满
    gpus = tf. config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    auto_search()