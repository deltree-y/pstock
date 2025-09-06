import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class BinManager:
    """
    用于分组边界(bins)的生成、保存、读取、分组标记与分箱分配的工具类(仅处理 json 文件)。
    支持自定义保存路径、分箱统计、异常处理、分箱分布可视化、只读属性保护等功能。
    初始化时要求输入数据(一维数组)或文件名(json),如输入文件名则自动读取分组,如输入数据则自动分组并生成分界点,并保存到指定 json 文件.
    """

    def __init__(self, data_or_filename, n_bins=None, save_path="bins.json"):
        """
        初始化,要求输入数据或 json 文件名.
        参数:
            data_or_filename: 一维数值型数组/列表 或 json 文件名(str)
            n_bins: 分组数量(int),仅当输入数据时需要指定
            save_path: 保存分组数据的 json 文件路径(默认 bins.json)
        行为:
            - 若输入为字符串且为文件名,自动从 json 文件读取分组信息,刷新 bins 和 n_bins
            - 若输入为数组/列表,自动生成分组分界点,刷新 bins 和 n_bins,并保存为 save_path
        """
        self._bins = None
        self._n_bins = None

        if isinstance(data_or_filename, str) and os.path.isfile(data_or_filename):
            with open(data_or_filename) as f:
                bins = np.array(json.load(f))
            self._validate_bins(bins)
            self._bins = bins
            self._n_bins = len(bins) - 1
        elif isinstance(data_or_filename, (list, np.ndarray, pd.Series)):
            if n_bins is None:
                raise ValueError("初始化时输入数据需指定分组数量 n_bins.")
            if n_bins >= len(data_or_filename):
                raise ValueError("分组数量 n_bins 不能大于或等于数据长度.")
            _, bins = pd.qcut(data_or_filename, q=n_bins, retbins=True, duplicates='drop')
            self._validate_bins(bins)
            self._bins = bins
            self._n_bins = len(bins) - 1
            self.save_bins_json(save_path)
        else:
            raise ValueError("输入必须为一维数据(list/numpy.ndarray/pandas.Series)或json文件名(str).")

    @property
    def bins(self):
        return self._bins.copy() if self._bins is not None else None

    @property
    def prop_bins(self):
        return self._bins.copy()[1:-1] if self._bins is not None else None

    @property
    def n_bins(self):
        return self._n_bins

    def _validate_bins(self, bins):
        # 检查分组边界有效性
        if len(bins) < 2:
            raise ValueError("bins 至少需要2个分界点.")
        if not np.all(np.diff(bins) > 0):
            raise ValueError("bins 必须严格递增.")

    def assign_bins(self, data, labels=False, right=True):
        """
        根据成员变量 self.bins,将数据分配到分箱.
        参数:
            data: 一维数值型数组或列表
            labels: 是否返回分组区间标签(True),否则返回组号(False)
            right: 区间是否右闭(默认为True)
        返回:
            若 labels=False,返回每个样本的组号(int);labels=True,返回每个样本的区间(pandas.Interval)
        """
        if self._bins is None:
            raise ValueError("请先生成或读取 bins.")
        return pd.cut(data, bins=self._bins, labels=False if not labels else None, right=right, include_lowest=True)

    def get_bin_labels(self, right=True):
        """
        获取分箱的区间标签.
        参数:
            right: 区间是否右闭(默认为True)
        返回:
            pandas.IntervalIndex,每个分组对应的区间
        """
        if self._bins is None:
            raise ValueError("bins 为空,请先生成分界点.")
        return pd.IntervalIndex.from_breaks(self._bins, closed='right' if right else 'left')

    def save_bins_json(self, filepath):
        """
        将 bins 保存为 json 文件.
        参数:
            filepath: 文件路径
        """
        if self._bins is None:
            raise ValueError("bins 为空,请先生成分界点.")
        with open(filepath, 'w') as f:
            json.dump(list(self._bins), f)

    def bin_summary(self, data):
        """
        返回每个分箱的样本数统计.
        参数:
            data: 一维数值型数组或列表
        返回:
            DataFrame: interval(区间)和 count(样本数)
        """
        bin_result = self.assign_bins(data)
        counts = pd.Series(bin_result).value_counts(sort=False)
        intervals = self.get_bin_labels()
        # 防止分箱区间与计数不一致
        count_list = [counts.get(i, 0) for i in range(len(intervals))]
        return pd.DataFrame({'interval': intervals, 'count': count_list})

    def plot_bins(self, data, show=True, save_path=None):
        """
        绘制分箱分布直方图.
        参数:
            data: 一维数值型数组或列表
            show: 是否显示图像(默认True)
            save_path: 若指定则保存图片
        """
        if self._bins is None:
            raise ValueError("bins 为空,请先生成分界点.")
        plt.hist(data, bins=self._bins, edgecolor='black', alpha=0.7)
        plt.title("Bin Distribution")
        plt.xlabel("Value")
        plt.ylabel("Count")
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    # 只读保护:禁止 bins 和 n_bins 被外部直接赋值
    @bins.setter
    def bins(self, value):
        raise AttributeError("bins 属性为只读,不允许直接赋值.")

    @n_bins.setter
    def n_bins(self, value):
        raise AttributeError("n_bins 属性为只读,不允许直接赋值.")

    def plot_bin_feature_correlation(self, bin_labels, feature_data, feature_names=None, show=True, save_path=None):
        """
        绘制分箱与特征均值的相关性热力图
        参数:
            bin_labels: 一维分箱编号数组（如 [0,1,1,2,...]），长度为样本数
            feature_data: 二维特征数组（shape: [样本数, 特征数]）
            feature_names: 特征名列表（如 ['close', 'volume', ...]）
            show: 是否显示图像
            save_path: 保存路径
        """
        import numpy as np
        import pandas as pd

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(feature_data.shape[1])]
        df = pd.DataFrame(feature_data, columns=feature_names)
        df['bin'] = bin_labels

        # 统计每个分箱的各特征均值
        mean_df = df.groupby('bin').mean().T

        plt.figure(figsize=(max(10, feature_data.shape[1]//2), 6))
        sns.heatmap(mean_df, annot=True, fmt=".2f", cmap='viridis')
        plt.title('Feature Mean by Bin')
        plt.xlabel('Bin')
        plt.ylabel('Feature')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()