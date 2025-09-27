# coding=utf-8
import os, logging
import numpy as np
from sklearn.utils import compute_class_weight
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.lstmmodel import LSTMModel
from model.residual_lstm import ResidualLSTMModel
from model.residual_tcn import ResidualTCNModel
from model.transformer import TransformerModel
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST, BASE_DIR, MODEL_DIR, NUM_CLASSES
from utils.utils import PredictType
from utils.utils import setup_logging, print_ratio
from predicproc.analyze import plot_confusion_by_model, print_predict_result, print_recall_score
from model.utils import  get_hard_samples, get_sample_weights

if __name__ == "__main__":
    setup_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # ================== 数据集准备 ==================
    si = StockInfo(TOKEN)
    t_list = (si.get_trade_open_dates('20250801', '20250829'))['trade_date'].tolist()
    primary_stock_code = '600036.SH'
    index_code_list = ['000001.SH']#,'399001.SZ']
    related_stock_list = ALL_CODE_LIST
    predict_type = PredictType.BINARY_T1L_L10  #二分类预测 T1 low <= -1.0%

    # 注意：train_size 设大容易使验证集过少，这里保持 0.9 但你可以回退到 0.85 观察稳定性
    ds = StockDataset(ts_code=primary_stock_code, idx_code_list=index_code_list, rel_code_list=related_stock_list, si=si,
                      start_date='20190104',
                      end_date='20250903',
                      train_size=0.9,
                      if_use_all_features=False,
                      predict_type=predict_type)

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    ty, vy = ty[:, 0], vy[:, 0]

    # ================== 模型选择 ==================
    #model_type = 'transformer'  
    model_type = 'residual_tcn'  
    #model_type = 'residual_lstm'
    #model_type = 'mini'

    # 循环训练调试参数
    for ns in [4]:
    #for l2 in [0.00001, 0.0001, 0.001, 0.01]:
    #for dp,bu in zip([4,4,4,6,6,6,8,8,8],[16,32,64,16,32,64,16,32,64]):
    #for lr in [0.001]:
        # ================== 训练参数 ==================
        epochs = 120
        batch_size = 1024
        learning_rate = 0.001#lr
        patience = 30
        p = 2
        dropout_rate = 0.3
        l2_reg = 0.00001#l2
        loss_type = 'binary_crossentropy'#'focal_loss'#'cross_entropy'#'weighted_cross_entropy'#'binary_crossentropy'
        hard_threshold = 0.4  #预测置信度低于此阈值的样本视为 hard 样本
        n_repeat = 3
        
        # 残差LSTM模型参数
        depth = 6
        base_units = 32
        # Transformer模型参数
        d_model = 256
        num_heads = 4
        ff_dim = 512
        num_layers = 4#nl#4
        # TCN模型参数
        nb_stacks = ns  #增大nb_stacks会整体重复残差结构，直接增加模型深度
        dilations = [1, 2, 4, 8, 16, 32]    #[1, 2, 4, 8] #每个stack内的dilation设置，增大dilation可以让模型看到更长时间的历史
        nb_filters = 64 #有多少组专家分别提取不同类型的特征
        kernel_size = 8 #每个专家一次能看到多长时间的历史窗口
        # LSTM Mini模型参数
        # 直接使用默认参数

        print(f"\n{'='*20} 开始训练: model={model_type}, epochs={epochs}, batch={batch_size}, lr={learning_rate} {'='*20}\n")        
        # ================== 根据上面的选择和参数自动配置模型参数 ==================
        if predict_type.is_classify():
            # 显著降低类别0权重，提高类别4权重
            class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=ty)
            #class_weights[0] *= 0.3
            #class_weights[4] *= 5
            cls_weights = dict(enumerate(class_weights))
            #cls_weights = dict(enumerate([0.46946348, 0.97702727, 1.18450308, 1.18450308, 1.18450308]))
        else:
            cls_weights = None

        save_path = os.path.join(BASE_DIR, MODEL_DIR, f"{primary_stock_code}_{model_type}_{predict_type}_ep{epochs}_bs{batch_size}.h5")
        # ================== 模型选择 ==================
        if model_type == 'residual_lstm':
            model = ResidualLSTMModel(
                x=tx, y=ty, test_x=vx, test_y=vy, p=p, depth=depth, base_units=base_units, dropout_rate=dropout_rate, use_se=True, se_ratio=8, l2_reg=l2_reg, class_weights=cls_weights, loss_type=loss_type, predict_type=predict_type
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                x=tx, y=ty, test_x=vx, test_y=vy, p=p, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, num_layers=num_layers, l2_reg=l2_reg, use_pos_encoding=True, use_gating=True, class_weights=cls_weights, loss_type=loss_type, predict_type=predict_type
            )
        elif model_type == 'residual_tcn':
            model = ResidualTCNModel(
                x=tx, y=ty, test_x=vx, test_y=vy, p=p, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,dilations=dilations, dropout_rate=dropout_rate, class_weights=cls_weights, loss_type=loss_type, l2_reg=l2_reg, causal=True, predict_type=predict_type
            )
        elif model_type == 'mini':
            model = LSTMModel(
                x=tx, y=ty, test_x=vx, test_y=vy, p=p, dropout_rate=dropout_rate, predict_type=predict_type
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"Class weights: {cls_weights}")
        #print(f"\nbins1: {ds.bins1.bins}\nbins2: {ds.bins2.bins}")
        print_ratio(ty, "ty")

        # ================== 训练 ==================
        # 1. 正常训练
        logging.info(f"\n1. 正常训练: tx shape: {tx.shape}, ty shape: {ty.shape}, vx shape: {vx.shape}, vy shape: {vy.shape}")
        train_ret = model.train(tx=tx, ty=ty, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        print_predict_result(t_list, ds, model, predict_type)
        vx_pred_raw = model.model.predict(vx)
        print_recall_score(vx_pred_raw, vy, predict_type)

        #plot_confusion_by_model(vx_pred_raw, vy1, num_classes=NUM_CLASSES, title=f"1. 正常训练 Confusion Matrix")
        
        # ================== 重点训练 ==================
        if False:
            # 2. 获取 hard 样本
            logging.info("\n2. 获取hard样本\增强样本\hard权重")
            tx_pred_raw = model.model.predict(tx)
            hard_x, hard_y = get_hard_samples(tx, ty, tx_pred_raw, predict_type, threshold=hard_threshold)
            aug_x, aug_y = ds.time_series_augmentation_4x(hard_x, hard_y, noise_level=0.01)
            if predict_type.is_classify():
                hard_mask = np.max(tx_pred_raw, axis=1) < hard_threshold
            elif predict_type.is_binary():
                hard_mask = np.abs(tx_pred_raw[:,0]-0.5)*2 < hard_threshold
            model.class_weight_dict = dict(enumerate(get_sample_weights(ty, hard_mask)))

            # 3. 增权/增强/多次训练任选其一或多种组合
            if True:
                # 3.1 增权训练(先找出预测置信度低的样本，给它们更高的权重)
                logging.info(f"\n3.1 增权训练: tx shape: {tx.shape}, ty shape: {ty.shape}, vx shape: {vx.shape}, vy shape: {vy.shape}")
                train_ret = model.train(tx=tx, ty=ty, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                print_predict_result(t_list, ds, model, predict_type)
                vx_pred_raw = model.model.predict(vx)
                print_recall_score(vx_pred_raw, vy, predict_type)
                #plot_confusion_by_model(model, vx, vy, num_classes=NUM_CLASSES, title=f"3.1 增权训练: Confusion Matrix")

            if False:
                # 3.2 数据增强训练
                logging.info(f"\n3.2 数据增强训练: tx shape: {aug_x.shape}, ty shape: {aug_y.shape}, vx shape: {vx.shape}, vy shape: {vy.shape}")
                model.train(tx=aug_x, ty=aug_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                print_predict_result(t_list, ds, model, predict_type)
                vx_pred_raw = model.model.predict(vx)
                print_recall_score(vx_pred_raw, vy, predict_type)
                plot_confusion_by_model(model, vx, vy, num_classes=NUM_CLASSES, title=f"3.3 多次训练: Confusion Matrix")

            if True:
                # 3.3 多次训练（可以和增权、增强结合）
                logging.info(f"\n3.3 多次训练: tx shape: {hard_x.shape}, ty shape: {hard_y.shape}, vx shape: {vx.shape}, vy shape: {vy.shape}")
                new_lr = learning_rate * 0.5
                patience *= 2
                batch_size = max(256, batch_size // 8)
                for _ in range(n_repeat):
                    model.train(tx=hard_x, ty=hard_y, epochs=epochs, batch_size=batch_size, learning_rate=new_lr, patience=patience)
                print_predict_result(t_list, ds, model, predict_type)
                vx_pred_raw = model.model.predict(vx)
                print_recall_score(vx_pred_raw, vy, predict_type)
                #plot_confusion_by_model(model, vx, vy, num_classes=NUM_CLASSES, title=f"3.3 多次训练: Confusion Matrix")

            # 或者将增强后的 hard 样本拼回主数据集再训练
            if False:
                # 3.4 将增强后的 hard 样本拼回主数据集再训练
                final_train_x = np.concatenate([tx, aug_x])
                final_train_y = np.concatenate([ty, aug_y])
                logging.info(f"\n3.4 将增强后的 hard 样本拼回主数据集再训练: tx shape: {final_train_x.shape}, ty shape: {final_train_y.shape}, vx shape: {vx.shape}, vy shape: {vy.shape}")
                model.train(tx=final_train_x, ty=final_train_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                print_predict_result(t_list, ds, model, predict_type)
                vx_pred_raw = model.model.predict(vx)
                print_recall_score(vx_pred_raw, vy, predict_type)
                plot_confusion_by_model(model, vx, vy, num_classes=NUM_CLASSES, title=f"3.4 将增强后的 hard 样本拼回主数据集再训练: Confusion Matrix")

        # ================== 评估 ==================
        #y_pred = model.model.predict(vx, batch_size=2048).reshape(-1)

        # ================== 保存模型 ==================
        model.save(save_path)
