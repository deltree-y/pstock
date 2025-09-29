#基于给定的日期list,使用给定的数据集和模型进行预测并打印结果
from predicproc.predict import Predict


def print_predict_result(t_list, ds, m, predict_type):
    for t0 in t_list:
        data, bp = ds.get_predictable_dataset_by_date(t0)
        #print("*************************************************************")
        #print(f"raw data is {data}")
        #print("*************************************************************\n")
        pred_scaled = m.model.predict(data, verbose=0)
        print(f"T0[{t0}]", end="")
        Predict(pred_scaled, bp, predict_type, ds.bins1, ds.bins2).print_predict_result()
    print("-------------------------------------------------------------")

