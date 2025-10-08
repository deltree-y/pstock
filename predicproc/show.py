#基于给定的日期list,使用给定的数据集和模型进行预测并打印结果
from predicproc.predict import Predict


def print_predict_result(t_list, ds, m, predict_type):
    print("-"*80)
    correct_cnt = 0
    for t0 in t_list:
        data, bp = ds.get_predictable_dataset_by_date(t0)
        real_y = ds.get_real_y_by_date(t0)
        raw_y = ds.get_raw_y_by_date(t0)
        pred_scaled = m.model.predict(data, verbose=0)
        pred = Predict(pred_scaled, bp, predict_type, ds.bins1, ds.bins2)
        print(f"T0[{t0}], raw_y:[{raw_y[0]*bp+bp:<.2f}], ", end="")
        pred.print_predict_result_with_real(real_y)
        correct_cnt += 1 if pred.pred_label==real_y[0,0] else 0 #仅分类和二分类任务有效#TODO:回归任务需要计算误差
    print(f"正确率: {correct_cnt/len(t_list):.2%}, 正确个数: {correct_cnt}/{len(t_list)}")
    print("-"*80)
