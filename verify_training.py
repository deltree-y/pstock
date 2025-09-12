# coding: utf-8
"""
verify_training.py (modified to support window-level volatility normalization)

主要变更：
- 新增命令行参数 --use_vol_norm，启用 StockDataset 中预计算的 window-level volatility normalization 目标。
- 当 --use_vol_norm 启用时：
    * 训练/验证 target 使用 ds.train_y_vol_norm_scaled / ds.test_y_vol_norm_scaled（z-score 后的 target/vol_pct）
    * inverse_fn 将把模型输出(在 vol-norm z-score 空间) 逆变换回百分比空间 (pred_pct)
      — 若 test window 的 vol_pct 可用（长度匹配），使用其逐样本乘回；
      — 否则回退到训练集平均 vol_pct。
- 增加更多日志，打印原始 target 与 vol-normalized target 的分布（均值/标准差/分位数），便于对比 "塌缩" 前后差异。
- 保持原有 Rank-Gauss 支持（若 --use_vol_norm 与 --rank_gauss 同时指定，会优先使用 --use_vol_norm 并给出警告）。

用法示例：
    python verify_training.py --epochs 40 --use_vol_norm
    python verify_training.py --epochs 40 --rank_gauss
"""
import os, sys, logging, argparse
import numpy as np
from pathlib import Path
from datasets.stockinfo import StockInfo
from dataset import StockDataset
from model.residual_lstm import ResidualLSTMModel
from utils.tk import TOKEN
from utils.const_def import ALL_CODE_LIST
from utils.utils import setup_logging

o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(str(Path(__file__).resolve().parents[0]))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ========== Rank-Gauss 目标变换 (保留) ==========
class RankGaussTransformer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.fitted = False

    def fit(self, y):
        y = np.asarray(y).reshape(-1)
        self.y_sorted = np.sort(y)
        ranks = np.argsort(np.argsort(y)) + 1
        n = len(y)
        cdf = (ranks - 0.5) / n
        from scipy.stats import norm
        self.gauss_vals = norm.ppf(np.clip(cdf, self.eps, 1 - self.eps))
        self.gauss_sorted = np.sort(self.gauss_vals)
        self.fitted = True
        return self

    def transform(self, y):
        if not self.fitted:
            raise ValueError("RankGaussTransformer not fitted.")
        y = np.asarray(y).reshape(-1)
        ranks = np.argsort(np.argsort(y)) + 1
        n = len(y)
        cdf = (ranks - 0.5) / n
        from scipy.stats import norm
        return norm.ppf(np.clip(cdf, self.eps, 1 - self.eps))

    def inverse_transform(self, y_gauss):
        if not self.fitted:
            raise ValueError("RankGaussTransformer not fitted.")
        y_gauss = np.asarray(y_gauss).reshape(-1)
        g_min, g_max = self.gauss_sorted[0], self.gauss_sorted[-1]
        y_gauss_clip = np.clip(y_gauss, g_min, g_max)
        return np.interp(y_gauss_clip, self.gauss_sorted, self.y_sorted)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40, help="基线实验最大 epoch")
    ap.add_argument("--rank_gauss", action="store_true", help="是否对目标使用 Rank-Gauss 变换(再 z-score)")
    ap.add_argument("--use_vol_norm", action="store_true", help="是否使用 dataset 中的 window-level volatility normalization 目标")
    ap.add_argument("--primary", type=str, default="600036.SH", help="主股票代码")
    ap.add_argument("--train_size", type=float, default=0.9, help="训练集比例")
    ap.add_argument("--start_date", type=str, default="20180101")
    ap.add_argument("--end_date", type=str, default="20250903")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2, help="残差块数量(基线)")
    ap.add_argument("--base_units", type=int, default=32)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no_se", action="store_true", help="关闭 SE Block (基线)")
    return ap.parse_args()


def build_dataset(args):
    si = StockInfo(TOKEN)
    ds = StockDataset(args.primary, [], ALL_CODE_LIST, si,
                      start_date=args.start_date, end_date=args.end_date,
                      train_size=args.train_size)
    tx, ty, vx, vy = ds.normalized_windowed_train_x, ds.train_y, ds.normalized_windowed_test_x, ds.test_y
    ty_reg = ty[:, 0].astype(float)  # 百分比（dataset.get_binned_y 中乘以100）
    vy_reg = vy[:, 0].astype(float)
    return ds, tx, ty_reg, vx, vy_reg


def prepare_targets(ty_reg, vy_reg, use_rank_gauss):
    """
    (保留) 不使用 vol-norm 情况下的 target 准备（z-score 或 rank-gauss + z-score）
    返回: train_target, val_target, inverse_fn, mean_y, std_y, extra
    inverse_fn(pred_scaled) -> returns percentage (pred_pct)
    """
    if not use_rank_gauss:
        mean_y, std_y = np.mean(ty_reg), np.std(ty_reg) + 1e-9
        train_target = (ty_reg - mean_y) / std_y
        val_target = (vy_reg - mean_y) / std_y
        def inverse_fn(pred_scaled):
            return pred_scaled * std_y + mean_y
        return train_target, val_target, inverse_fn, mean_y, std_y, {}
    else:
        rgt = RankGaussTransformer().fit(ty_reg)
        ty_rg = rgt.transform(ty_reg)
        vy_rg = rgt.transform(vy_reg)
        mean_y, std_y = np.mean(ty_rg), np.std(ty_rg) + 1e-9
        train_target = (ty_rg - mean_y) / std_y
        val_target = (vy_rg - mean_y) / std_y
        def inverse_fn(pred_scaled):
            gauss_val = pred_scaled * std_y + mean_y
            return rgt.inverse_transform(gauss_val)
        return train_target, val_target, inverse_fn, mean_y, std_y, {"rank_gauss": True}


def run_single_experiment(desc,
                          tx, train_target,
                          vx, val_target,
                          vy_original,
                          inverse_fn,
                          mean_y, std_y,
                          cfg):
    logging.info(f"[RUN] {desc} cfg={cfg}")
    model = ResidualLSTMModel(
        x=tx,
        y=train_target,
        test_x=vx,
        test_y=val_target,
        p=cfg["p"],
        depth=cfg["depth"],
        base_units=cfg["base_units"],
        dropout_rate=cfg["dropout"],
        use_se=cfg["use_se"],
        se_ratio=8,
        l2_reg=1e-5
    )
    model.train(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        patience=cfg["patience"]
    )
    # 预测 -> 逆标准化 -> (若 rank_gauss)逆 rank -> 真实空间
    pred_scaled = model.model.predict(vx, batch_size=2048).reshape(-1)
    pred_real = inverse_fn(pred_scaled)  # inverse_fn 必须返回 percentage 单位 (例如 1.23 表示 1.23%)
    # 计算指标（vy_original 是 percentage）
    real_mean, real_std = np.mean(vy_original), np.std(vy_original) + 1e-9
    pred_mean, pred_std = np.mean(pred_real), np.std(pred_real) + 1e-9
    collapse_ratio = pred_std / real_std
    direction_acc = np.mean(np.sign(pred_real) == np.sign(vy_original))
    pearson = np.corrcoef(pred_real, vy_original)[0, 1]
    try:
        from scipy.stats import spearmanr
        spearman = spearmanr(pred_real, vy_original).correlation
    except Exception:
        spearman = np.nan
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    real_q = np.quantile(vy_original, qs)
    pred_q = np.quantile(pred_real, qs)

    logging.info(f"[RESULT]{desc} real_mean={real_mean:.4f} real_std={real_std:.4f} "
                 f"pred_mean={pred_mean:.4f} pred_std={pred_std:.4f} "
                 f"collapse_ratio={collapse_ratio:.3f} dir_acc={direction_acc:.3f} "
                 f"pearson={pearson:.3f} spearman={spearman:.3f}")
    for qi, rq, pq in zip(qs, real_q, pred_q):
        logging.info(f"[RESULT]{desc} Q{qi*100:>4.0f} real={rq:.4f} pred={pq:.4f}")

    return {
        "desc": desc,
        "collapse_ratio": collapse_ratio,
        "direction_acc": direction_acc,
        "pearson": pearson,
        "spearman": spearman,
        "pred_std": pred_std,
        "real_std": real_std,
        "pred_mean": pred_mean,
        "real_mean": real_mean
    }


def main():
    setup_logging()
    args = parse_args()
    logging.info("=== STEP 1: 构建数据集 ===")
    ds, tx, ty_reg, vx, vy_reg = build_dataset(args)

    # 打印原始分布信息
    logging.info(f"[DATA] raw train target mean/std = {np.mean(ty_reg):.4f}/{np.std(ty_reg):.4f}, val std = {np.std(vy_reg):.4f}")
    logging.info(f"[DATA] train percentiles: {np.quantile(ty_reg, [0.01,0.1,0.25,0.5,0.75,0.9,0.99])}")

    # 优先处理 vol-normalization（若用户请求）
    if args.use_vol_norm:
        if not hasattr(ds, "train_y_vol_norm_scaled") or ds.train_y_vol_norm_scaled is None or ds.train_y_vol_norm_scaled.size == 0:
            logging.warning("Dataset does not contain vol-normalized targets; falling back to normal targets.")
            args.use_vol_norm = False
        else:
            logging.info("Using dataset-provided window-level volatility-normalized targets.")
            # ds.windowed_train_vol_pct / ds.windowed_test_vol_pct (百分比)
            logging.info(f"[VOL] train vol pct mean/std = {np.mean(ds.windowed_train_vol_pct):.4f}/{np.std(ds.windowed_train_vol_pct):.4f}")
            logging.info(f"[VOL] train y_vol_norm mean/std = {np.mean(ds.train_y_vol_norm):.4f}/{np.std(ds.train_y_vol_norm):.4f}")

            # 直接使用 dataset 中预计算好的 scaled targets 作为训练目标
            train_target = ds.train_y_vol_norm_scaled
            val_target = ds.test_y_vol_norm_scaled if hasattr(ds, "test_y_vol_norm_scaled") else np.array([])

            # 逆变换函数：从模型输出（vol-norm zscore 空间） -> 百分比空间
            def inverse_fn_vol(pred_scaled):
                # pred_scaled: z-scored (train_y_vol_norm_scaled)
                train_mean = ds.train_volnorm_mean
                train_std = ds.train_volnorm_std
                pred_volnorm = pred_scaled * train_std + train_mean  # target / vol_pct
                # multiply back by vol pct to get percentage predictions
                test_vol_pct = getattr(ds, "windowed_test_vol_pct", None)
                if test_vol_pct is not None and test_vol_pct.size == pred_volnorm.size:
                    pred_pct = pred_volnorm * test_vol_pct
                else:
                    mean_train_vol = np.mean(getattr(ds, "windowed_train_vol_pct", np.array([1.0])))
                    logging.warning("test vol pct unavailable or length mismatch -> using mean train vol for scaling back")
                    pred_pct = pred_volnorm * mean_train_vol
                return pred_pct

            inverse_fn = inverse_fn_vol
            mean_y_used = ds.train_volnorm_mean
            std_y_used = ds.train_volnorm_std

            # align lengths: ensure tx/train_target lengths match
            if len(train_target) != tx.shape[0]:
                minlen = min(len(train_target), tx.shape[0])
                logging.warning(f"[ALIGN] train target/x length mismatch, trimming to {minlen}")
                train_target = train_target[:minlen]
                tx = tx[:minlen]
            if vx is not None and vx.size > 0 and val_target is not None and val_target.size > 0 and len(val_target) != vx.shape[0]:
                minlen = min(len(val_target), vx.shape[0])
                logging.warning(f"[ALIGN] val target/x length mismatch, trimming to {minlen}")
                val_target = val_target[:minlen]
                vx = vx[:minlen]

    # 若不使用 vol-norm，则走原有流程（rank-gauss 或 z-score）
    if not args.use_vol_norm:
        logging.info("Preparing targets using z-score (or rank-gauss if requested).")
        train_target, val_target, inverse_fn, mean_y_used, std_y_used, extra = prepare_targets(ty_reg, vy_reg, args.rank_gauss)
        logging.info(f"Target transform: rank_gauss={args.rank_gauss}")

    # 基线配置
    base_cfg = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "p": args.p,
        "depth": args.depth,
        "base_units": args.base_units,
        "dropout": args.dropout,
        "lr": args.lr,
        "patience": args.patience,
        "use_se": not args.no_se
    }

    logging.info("=== STEP 3: 运行基线实验 ===")
    baseline_result = run_single_experiment(
        desc="baseline",
        tx=tx,
        train_target=train_target,
        vx=vx,
        val_target=val_target,
        vy_original=vy_reg,
        inverse_fn=inverse_fn,
        mean_y=mean_y_used,
        std_y=std_y_used,
        cfg=base_cfg
    )

    collapse_threshold = 0.35
    results = [baseline_result]

    # 若塌缩，尝试一组对策配置
    if baseline_result and baseline_result["collapse_ratio"] < collapse_threshold:
        logging.info("=== 检测到预测方差塌缩，开始对策网格实验 ===")
        variant_cfgs = [
            ("lower_dropout", {**base_cfg, "dropout": 0.10}),
            ("higher_lr", {**base_cfg, "lr": 2e-3}),
            ("no_se", {**base_cfg, "use_se": False}),
            ("lr_dropout_no_se", {**base_cfg, "lr": 2e-3, "dropout": 0.10, "use_se": False}),
        ]
        for tag, cfg in variant_cfgs:
            res = run_single_experiment(
                desc=tag,
                tx=tx,
                train_target=train_target,
                vx=vx,
                val_target=val_target,
                vy_original=vy_reg,
                inverse_fn=inverse_fn,
                mean_y=mean_y_used,
                std_y=std_y_used,
                cfg=cfg
            )
            results.append(res)

    # 汇总打印
    logging.info("=== SUMMARY ===")
    for r in results:
        logging.info(f"{r['desc']}: collapse_ratio={r['collapse_ratio']:.3f} "
                     f"pred_std={r['pred_std']:.4f} real_std={r['real_std']:.4f} "
                     f"pearson={r['pearson']:.3f} dir_acc={r['direction_acc']:.3f}")

    logging.info("完成。你可以基于 collapse_ratio 与 pearson 挑选进一步调参方向。")
    logging.info("若所有配置仍塌缩，可尝试：减少特征 / 加强正则 / 更换目标定义 / 引入差分或 log-return。")


if __name__ == "__main__":
    main()