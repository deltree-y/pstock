#基于给定的日期list,使用给定的数据集和模型进行预测并打印结果
from predicproc.predict import Predict, PredictType
import numpy as np
from dataset import StockDataset


def print_predict_result(t_list, ds:StockDataset, m, predict_type:PredictType, threshold=0.5):
    print("-"*80)
    correct_cnt = 0
    predict_wrong_list_str = ""
    correct_probs = []
    wrong_probs = []
    residual = []
    pred_values = []
    low_high_symbol = "涨" if predict_type.is_high() else "跌"

    print(f"\n预测结果如下:")
    for t0 in t_list:
        _, data, bp = ds.get_predictable_dataset_by_date(t0)
        real_y = ds.get_real_y_by_date(t0)
        #raw_y = ds.get_raw_y_by_date(t0)
        pred_scaled = m.model.predict(data, verbose=0)
        pred = Predict(pred_scaled, bp, predict_type, ds.bins1, ds.bins2, threshold=threshold)

        pred_dot, predict_wrong_str = pred.get_predict_result_with_real_str(t0, bp, real_y)
        #predict_wrong_list_str += f"T0[{t0}], 真实/预测(差异)_{low_high_symbol} : [{(real_y*bp+bp):<.2f}/{predict_wrong_str}\n" if predict_wrong_str!="" else ""
        predict_wrong_list_str += f"{predict_wrong_str}\n" if predict_wrong_str!="" else ""

        if pred.is_binary:
            is_correct = (pred.pred_label == real_y)
        elif pred.is_classify:
            is_correct = (pred.y1r.get_label() == real_y)
        elif pred.is_regression:
            is_correct = (pred_dot == "o")
            residual.append(abs(pred.pred_value - real_y*100))
            pred_values.append(pred.pred_value)
        else:
            print("未知的预测类型，无法判断正确性。")
            is_correct = False

        if is_correct:
                correct_cnt += 1

        # 统计置信率
        prob_rate = None
        if pred.is_binary:
            #prob_rate = pred.prob# if pred.pred_label == 1 else (1 - pred.prob)
            prob_rate = abs(pred.prob-0.5)*2
        elif pred.is_classify:
            prob_rate = max(pred.predicted_data[0])
        # 回归类型不统计平均置信率
        if prob_rate is not None:
            if is_correct:
                correct_probs.append(prob_rate)
            else:
                wrong_probs.append(prob_rate)

        print(f"{pred_dot}", end="", flush=True)

    print(f"\n\n预测错误列表:\n{predict_wrong_list_str}")
    print(f"正确率: {correct_cnt/len(t_list):.2%}, 正确个数: {correct_cnt}/{len(t_list)}")
    if correct_probs:
        print(f"正确平均置信率: {np.mean(correct_probs)*100:.2f}%")
    else:
        print("无正确数据统计平均置信率")
    if wrong_probs:
        print(f"错误平均置信率: {np.mean(wrong_probs)*100:.2f}%")
    else:
        print("无错误数据统计平均置信率")
    print("-"*100)

    return correct_cnt / len(t_list), np.mean(correct_probs)*100, np.mean(wrong_probs)*100, np.mean(residual) if residual else None, np.std(pred_values) if pred_values else None


def print_predict_result_soft_gated_t1l10(t_list, ds_gate:StockDataset, m_gate, ds_reg:StockDataset, m_reg, y_base=-0.2, gamma=1.0):
    """
    二阶段软门控预测：使用 BINARY_T1L10 输出的概率 p 对 REGRESS_T1L 的回归输出 y_reg 进行软门控融合。
    
    融合公式：
        p_g = p ** gamma
        y_hat = p_g * y_reg + (1 - p_g) * y_base
    
    参数说明：
        t_list: 要预测的日期列表
        ds_gate: 二分类数据集 (BINARY_T1L10)
        m_gate: 二分类模型 (BINARY_T1L10)
        ds_reg: 回归数据集 (REGRESS_T1L)
        m_reg: 回归模型 (REGRESS_T1L)
        y_base: 基础回归值（百分点），默认 -0.2 表示 -0.2%
        gamma: 锐化参数，默认 1.0（不锐化）
    
    返回值：
        (正确率, None, None, 平均残差, 预测值标准差)
        其中第2、3项为None，保持与 print_predict_result 返回结构一致（不统计置信率）
    
    单位说明：
        - p: 二分类模型输出的概率 [0, 1]
        - y_reg: 回归模型输出的百分点变化率（例如 -0.5 表示 -0.5%）
        - y_base: 百分点单位（例如 -0.2 表示 -0.2%）
        - y_hat: 融合后的预测值，百分点单位
        - real_y: 真实值为倍率（例如 -0.005 表示 -0.5%），需乘以100转为百分点
    """
    print("-"*80)
    correct_cnt = 0
    predict_wrong_list_str = ""
    residual = []
    pred_values = []
    
    # 使用回归数据集的 PredictType 来构造 Predict 对象
    predict_type = ds_reg.predict_type
    
    print(f"\n软门控预测结果如下 (y_base={y_base}, gamma={gamma}):")
    for t0 in t_list:
        try:
            # 1. 获取二分类门控概率 p
            _, data_gate, bp_gate = ds_gate.get_predictable_dataset_by_date(t0)
            pred_gate_scaled = m_gate.model.predict(data_gate, verbose=0)
            p = float(pred_gate_scaled[0, 0])  # BINARY_T1L10 输出概率
            
            # 2. 获取回归预测值 y_reg
            _, data_reg, bp_reg = ds_reg.get_predictable_dataset_by_date(t0)
            pred_reg_scaled = m_reg.model.predict(data_reg, verbose=0)
            y_reg = float(pred_reg_scaled[0, 0])  # REGRESS_T1L 输出百分点
            
            # 使用回归数据集的基准价（通常两个数据集的bp应该一致）
            bp = bp_reg
            
            # 3. 软门控融合
            p_g = p ** gamma
            y_hat = p_g * y_reg + (1 - p_g) * y_base
            
            # 4. 获取真实值（从回归数据集获取，确保一致性）
            real_y = ds_reg.get_real_y_by_date(t0)
            
            # 5. 构造 Predict 对象以复用现有的输出格式
            # 将 y_hat 包装成模型输出格式 [1, 1]
            pred_data = np.array([[y_hat]])
            pred = Predict(pred_data, bp, predict_type)
            
            # 6. 获取预测结果字符串
            pred_dot, predict_wrong_str = pred.get_predict_result_with_real_str(t0, bp, real_y)
            predict_wrong_list_str += f"{predict_wrong_str}\n" if predict_wrong_str != "" else ""
            
            # 7. 判断正确性和统计
            is_correct = (pred_dot == "o")
            if is_correct:
                correct_cnt += 1
            
            residual.append(abs(y_hat - real_y*100))
            pred_values.append(y_hat)
            
            print(f"{pred_dot}", end="", flush=True)
            
        except Exception as e:
            print(f"\n日期 {t0} 处理失败: {e}")
            print(f"x", end="", flush=True)
            continue
    
    print(f"\n\n预测错误列表:\n{predict_wrong_list_str}")
    print(f"正确率: {correct_cnt/len(t_list):.2%}, 正确个数: {correct_cnt}/{len(t_list)}")
    if residual:
        print(f"平均残差(MAE): {np.mean(residual):.2f} 百分点")
    if pred_values:
        print(f"预测值标准差: {np.std(pred_values):.2f} 百分点")
    print("-"*100)
    
    # 返回结构与 print_predict_result 保持一致
    # (正确率, 正确平均置信率, 错误平均置信率, 平均残差, 预测值标准差)
    # 软门控不统计置信率，用 None 填充
    return correct_cnt / len(t_list), None, None, np.mean(residual) if residual else None, np.std(pred_values) if pred_values else None
