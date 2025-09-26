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
    primary_stock_code = '600036.SH'
    t_list = ['20250829', '20250901', '20250902', '20250903']
    index_code_list = []
    related_stock_list = ALL_CODE_LIST
    predict_type = PredictType.BINARY_T1L_L05  #二分类预测 T1 low <= -0.5%
    #predict_type = PredictType.CLASSIFY  #多分类预测 T1 low 分箱类别

    # 注意：train_size 设大容易使验证集过少，这里保持 0.9 但你可以回退到 0.85 观察稳定性
    ds = StockDataset(ts_code=primary_stock_code,
                      idx_code_list=index_code_list,
                      rel_code_list=related_stock_list,
                      si=si,
                      start_date='20190104',
                      end_date='20250903',
                      train_size=0.9,
                      if_use_all_features=False,
                      predict_type=predict_type)

    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    
    #仅取一次预测
    ty1 = ty[:, 0]
    vy1 = vy[:, 0]

    # ================== 模型选择 ==================
    #model_type = 'transformer'  # <<< 修改这里即可切换模型
    #model_type = 'residual_tcn'  # <<< 修改这里即可切换模型
    model_type = 'residual_lstm'
    #model_type = 'mini'

    # 循环训练调试参数
    #for para1,para2 in zip([4,4,4,6,6,6,8,8,8],[16,32,64,16,32,64,16,32,64]):
    for lr in [0.002]:
        # ================== 训练参数 ==================
        n_repeat = 3
        epochs = 100
        batch_size = 2048
        learning_rate = lr
        patience = 20
        p = 2
        dropout_rate = 0.3
        l2_reg = 0.0001
        loss_type = 'focal_loss'#'focal_loss'#'cross_entropy'#'weighted_cross_entropy'
        
        if predict_type == PredictType.CLASSIFY:
            # 显著降低类别0权重，提高类别4权重
            class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=ty1)
            #class_weights[0] *= 0.3
            #class_weights[4] *= 5
            cls_weights = dict(enumerate(class_weights))
            #cls_weights = dict(enumerate([0.46946348, 0.97702727, 1.18450308, 1.18450308, 1.18450308]))
        else:
            cls_weights = None

        save_path = os.path.join(BASE_DIR, MODEL_DIR, f"{primary_stock_code}_{model_type}_{predict_type}_ep{epochs}_bs{batch_size}.h5")
        # ================== 模型选择 ==================
        if model_type == 'residual_lstm':
            # 残差LSTM模型参数
            depth = 6
            base_units = 32
            use_se = True

            model = ResidualLSTMModel(
                x=tx, y=ty1, test_x=vx, test_y=vy1, p=p,
                depth=depth, base_units=base_units, dropout_rate=dropout_rate, use_se=use_se,
                se_ratio=8, l2_reg=l2_reg,
                class_weights=cls_weights, loss_type=loss_type,
                predict_type=predict_type
            )
        elif model_type == 'transformer':
            # Transformer模型参数
            d_model = 256
            num_heads = 4
            ff_dim = 512
            num_layers = 4

            model = TransformerModel(
                x=tx, y=ty1, test_x=vx, test_y=vy1, p=p,
                d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, num_layers=num_layers,
                l2_reg=l2_reg, use_pos_encoding=True, use_gating=True,
                class_weights=cls_weights, loss_type = loss_type
            )
        elif model_type == 'residual_tcn':
            # TCN模型参数
            dilations = [1, 2, 4, 8, 16, 32]
            nb_filters = 64
            kernel_size = 8
            nb_stacks = 2
            causal = True  # 因为是时间序列，建议使用因果卷积

            model = ResidualTCNModel(
                x=tx, y=ty1, test_x=vx, test_y=vy1, p=p,
                nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                dilations=dilations, dropout_rate=dropout_rate,
                class_weights=cls_weights, loss_type = loss_type,
                l2_reg=l2_reg, causal=causal
            )
        elif model_type == 'mini':
            # LSTM Mini模型参数
            model = LSTMModel(
                x=tx, y=ty1, test_x=vx, test_y=vy1, p=p,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"Class weights: {cls_weights}")
        print(f"\nbins1: {ds.bins1.bins}\nbins2: {ds.bins2.bins}")
        print_ratio(ty1, "ty1")

        # ================== 训练 ==================
        # 1. 正常训练
        logging.info(f"1. 正常训练: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
        logging.info(f"tx shape: {tx.shape}, ty1 shape: {ty1.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
        train_ret = model.train(tx=tx, ty=ty1, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
        print_predict_result(t_list, ds, model, predict_type)
        vx_pred_raw = model.model.predict(vx)
        print_recall_score(vx_pred_raw, vy1, predict_type)

        #plot_confusion_by_model(vx_pred_raw, vy1, num_classes=NUM_CLASSES, title=f"1. 正常训练 Confusion Matrix")
        
        # ================== 重点训练 ==================
        if False:
            # 2. 获取 hard 样本
            logging.info("2. 获取 hard 样本")
            hard_x, hard_y = get_hard_samples(tx, ty1, model.model, threshold=0.3)
            aug_x, aug_y = ds.time_series_augmentation_4x(hard_x, hard_y, noise_level=0.01)

            # 3. 增权/增强/多次训练任选其一或多种组合
            if True:
                # 3.1 增权训练(先找出预测置信度低的样本，给它们更高的权重)
                hard_mask = np.max(model.model.predict(tx), axis=1) < 0.3
                model.class_weight_dict = dict(enumerate(get_sample_weights(ty1, hard_mask)))
                logging.info(f"3.1 增权训练: tx shape: {tx.shape}, ty1 shape: {ty1.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
                train_ret = model.train(tx=tx, ty=ty1, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                #res_lstm.class_weight_dict = dict(enumerate([1,1,1,1,1,1]))  # 还原默认权重
                print_predict_result(t_list, ds, model)
                plot_confusion_by_model(model, vx, vy1, num_classes=NUM_CLASSES, title=f"3.1 增权训练: Confusion Matrix")

            if False:
                # 3.2 数据增强训练
                logging.info(f"3.2 数据增强训练: tx shape: {aug_x.shape}, ty1 shape: {aug_y.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
                model.train(tx=aug_x, ty=aug_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                print_predict_result(t_list, ds, model)
                plot_confusion_by_model(model, vx, vy1, num_classes=NUM_CLASSES, title=f"3.3 多次训练: Confusion Matrix")

            if True:
                # 3.3 多次训练（可以和增权、增强结合）
                logging.info(f"3.3 多次训练: tx shape: {hard_x.shape}, ty1 shape: {hard_y.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
                new_lr = learning_rate * 0.5
                for _ in range(n_repeat):
                    model.train(tx=hard_x, ty=hard_y, epochs=epochs, batch_size=batch_size, learning_rate=new_lr, patience=patience)
                print_predict_result(t_list, ds, model)
                plot_confusion_by_model(model, vx, vy1, num_classes=NUM_CLASSES, title=f"3.3 多次训练: Confusion Matrix")

            # 或者将增强后的 hard 样本拼回主数据集再训练
            if False:
                # 3.4 将增强后的 hard 样本拼回主数据集再训练
                final_train_x = np.concatenate([tx, aug_x])
                final_train_y = np.concatenate([ty1, aug_y])
                logging.info(f"3.4 将增强后的 hard 样本拼回主数据集再训练: tx shape: {final_train_x.shape}, ty1 shape: {final_train_y.shape}, vx shape: {vx.shape}, vy1 shape: {vy1.shape}")
                model.train(tx=final_train_x, ty=final_train_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
                print_predict_result(t_list, ds, model)
                plot_confusion_by_model(model, vx, vy1, num_classes=NUM_CLASSES, title=f"3.4 将增强后的 hard 样本拼回主数据集再训练: Confusion Matrix")

        # ================== 评估 ==================
        #y_pred = model.model.predict(vx, batch_size=2048).reshape(-1)

        # ================== 保存模型 ==================
        model.save(save_path)
