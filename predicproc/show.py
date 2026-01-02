#基于给定的日期list,使用给定的数据集和模型进行预测并打印结果
from predicproc.predict import Predict
import numpy as np
from dataset import StockDataset


def print_predict_result(t_list, ds:StockDataset, m, predict_type, threshold=0.5):
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
        data, bp = ds.get_predictable_dataset_by_date(t0)
        real_y = ds.get_real_y_by_date(t0)
        #raw_y = ds.get_raw_y_by_date(t0)
        pred_scaled = m.model.predict(data, verbose=0)
        pred = Predict(pred_scaled, bp, predict_type, ds.bins1, ds.bins2, threshold=threshold)

        pred_dot, predict_wrong_str = pred.get_predict_result_with_real_str(t0, bp, real_y)
        #predict_wrong_list_str += f"T0[{t0}], 真实/预测(差异)_{low_high_symbol} : [{(real_y*bp+bp):<.2f}/{predict_wrong_str}\n" if predict_wrong_str!="" else ""
        predict_wrong_list_str += f"{predict_wrong_str}\n" if predict_wrong_str!="" else ""

        if pred.is_binary:
            is_correct = pred.pred_label == real_y
        elif pred.is_classify:
            is_correct = pred.y1r.get_label() == real_y
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
            prob_rate = pred.prob# if pred.pred_label == 1 else (1 - pred.prob)
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
