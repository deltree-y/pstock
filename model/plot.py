from matplotlib import pyplot as plt



    
def plot_regression_result(y_true, y_pred, title="回归预测结果", save_path=None):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="real", marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label="pred", marker='x', linestyle='--', alpha=0.7)
    plt.ylim(-8, 2)
    plt.title(title)
    plt.xlabel("sn")
    plt.ylabel("chg_pct(%)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_error_distribution(y_true, y_pred, title="mae/rmse distribution", save_path=None):
    errors = y_pred - y_true
    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("mae(pred-real,%)")
    plt.ylabel("sample count")
    plt.xlim(-5, 5)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
