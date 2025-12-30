# -*- coding: utf-8 -*-
import numpy as np
import logging

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return x

def check_xy_alignment_conv2d(ds, n_samples=8, seed=42, use_train=True, show_channels=3):
    """
    (2) 检查 Conv2D 多通道模式下 x-window 与 y 是否对齐

    依赖 ds.use_conv2_channel=True 且 ds.split_train_test_dataset_by_stock() 已保存:
      - ds.channel_train_x_list / ds.channel_test_x_list : list of ndarray，每个元素 shape [N, cols(含日期+特征+y)]
      - ds.train_y_no_window / ds.test_y_no_window
      - ds.raw_train_y / ds.raw_test_y  (未分箱/未二值化的原始 y：t1l,t1h,t2l,t2h)

    核心思想：
      - 在多通道模式下，y 取主通道(0)；
      - train/test 切分时每个通道都做了 test_count 切片；
      - 对齐检查就是：随机取一个 window 起点 i，
            window 覆盖的日期 = ch[0][i : i+window_size, 0]
            label 对应的日期 = ch[0][i, 0] （因为你的 y 是按当日 t0 生成的）
        然后打印 y[i]、raw_y[i]，确认这一天对应的是同一个 t0。
    """
    assert getattr(ds, "use_conv2_channel", False), "此函数只用于 use_conv2_channel=True 的 Conv2D 多通道模式"
    assert getattr(ds, "channel_train_x_list", None) is not None, "缺少 ds.channel_train_x_list，请确认 split_train_test_dataset_by_stock() 已在 conv2 模式下运行"

    ch_list = ds.channel_train_x_list if use_train else ds.channel_test_x_list
    y_no_window = ds.train_y_no_window if use_train else ds.test_y_no_window
    raw_y = ds.raw_train_y if use_train else ds.raw_test_y
    y_windowed = ds.train_y if use_train else ds.test_y

    main = ch_list[0]  # 主通道
    N = main.shape[0]
    W = ds.window_size
    n_windows = N - W + 1
    if n_windows <= 0:
        raise ValueError(f"数据不足以窗口化: N={N}, window={W}")

    rng = np.random.default_rng(seed)
    picks = rng.choice(n_windows, size=min(n_samples, n_windows), replace=False)

    logging.info("=" * 90)
    logging.info(f"[ALIGN CHECK] use_train={use_train}, main_N={N}, window={W}, windows={n_windows}")
    logging.info(f"[ALIGN CHECK] y_no_window shape={y_no_window.shape}, y_windowed shape={y_windowed.shape}, raw_y shape={raw_y.shape}")
    logging.info("=" * 90)

    for k, i in enumerate(picks, start=1):
        win_dates = main[i:i+W, 0]
        t0_date = main[i, 0]
        t_last = main[i+W-1, 0]

        # y 在 __init__ 里做过裁剪: y_windowed = y_no_window[:-W+1]
        y_label = y_windowed[i] if i < len(y_windowed) else None
        y_label_nw = y_no_window[i] if i < len(y_no_window) else None
        raw_y_i = raw_y[i] if i < len(raw_y) else None

        # 打印每个 channel 同一窗口的起止日期，确认通道是否对齐
        ch_date_ranges = []
        for ch_idx, ch in enumerate(ch_list[:show_channels]):
            ch_win = ch[i:i+W, 0]
            ch_date_ranges.append((ch_idx, _safe_int(ch_win[0]), _safe_int(ch_win[-1])))

        logging.info(f"[{k}/{len(picks)}] window_start_idx={i}")
        logging.info(f"  main_window_dates: {_safe_int(win_dates[0])} -> {_safe_int(t_last)} (t0={_safe_int(t0_date)})")
        logging.info(f"  channels(date_range): {ch_date_ranges}")
        logging.info(f"  y_no_window[i]={y_label_nw} | y_windowed[i]={y_label}")
        logging.info(f"  raw_y[i](t1l,t1h,t2l,t2h)={raw_y_i}")

        # 额外一致性检查：y_windowed 应等于 y_no_window 的前 n_windows 项
        if y_label is not None and y_label_nw is not None:
            if np.any(np.asarray(y_label) != np.asarray(y_label_nw)):
                logging.warning("  !!! y_windowed[i] != y_no_window[i]，存在裁剪/索引错位风险")

    logging.info("=" * 90)
    logging.info("提示：如果你发现 window 的 t0 日期与 y[i] 对应的真实涨跌幅日期不一致，基本可以断定 x/y 错位。")
    logging.info("=" * 90)


def calc_fill_ratio_raw_channels(ds, fill_val=-1.0, use_train=True, include_y_cols=True):
    """
    (3) 统计填充值/缺失比例：按 channel/feature 统计 fill_val(默认-1) 占比

    注意：
    - ds.channel_train_x_list / ds.channel_test_x_list 每行包含 [date, features..., y...]
    - 你可以选择 include_y_cols=False 只统计特征列（推荐）
    """
    assert getattr(ds, "use_conv2_channel", False), "此函数只用于 use_conv2_channel=True 的 Conv2D 多通道模式"
    ch_list = ds.channel_train_x_list if use_train else ds.channel_test_x_list
    assert ch_list is not None and len(ch_list) > 0

    y_cnt = getattr(ds, "y_cnt", 0)
    stats = []
    logging.info("=" * 90)
    logging.info(f"[FILL RATIO RAW] use_train={use_train}, fill_val={fill_val}, include_y_cols={include_y_cols}")
    logging.info(f"[FILL RATIO RAW] channels={len(ch_list)}, y_cnt={y_cnt}")
    logging.info("=" * 90)

    for ch_idx, ch in enumerate(ch_list):
        # ch shape: [N, cols]
        if include_y_cols:
            arr = ch[:, 1:]  # 去掉日期列
        else:
            arr = ch[:, 1:-y_cnt] if y_cnt > 0 else ch[:, 1:]

        arr = np.asarray(arr, dtype=float)
        total = arr.size
        fill = np.sum(arr == fill_val)
        ratio = fill / total if total > 0 else 0.0

        # 每个 feature 的 fill ratio
        per_feat = np.mean(arr == fill_val, axis=0) if arr.ndim == 2 and arr.shape[0] > 0 else None
        stats.append((ch_idx, ratio, per_feat))

        logging.info(f"channel[{ch_idx}] shape={arr.shape}, fill_ratio={ratio:.2%}")

        if per_feat is not None:
            top_idx = np.argsort(-per_feat)[:10]  # top10 最缺失特征
            top_str = ", ".join([f"f{j}:{per_feat[j]:.1%}" for j in top_idx])
            logging.info(f"  top missing features: {top_str}")

    logging.info("=" * 90)
    logging.info("提示：如果某些 channel 的 fill_ratio 很高(比如 >20%~30%)，Conv2D 很容易学到缺失模式并塌缩输出。")
    logging.info("=" * 90)
    return stats


def calc_fill_ratio_windowed_input(ds, fill_val=-1.0, use_train=True):
    """
    (3) 统计窗口化后、实际喂给 Conv2D 的输入里 fill_val 占比
    适用于 ds.normalized_windowed_train_x / test_x

    注意：
    - 这里统计的是“归一化后的输入张量”中仍等于 fill_val 的比例。
      如果你在归一化前是 -1，StandardScaler 后通常不会仍是 -1。
      所以这个函数更适合用来检查：你是否在训练前用 np.nan_to_num(..., nan=-1) 又把 NaN 变回 -1 等情况。
    """
    x = ds.normalized_windowed_train_x if use_train else ds.normalized_windowed_test_x
    if x is None:
        raise ValueError("x is None, 可能 train_size=1 导致没有 test_x")

    x = np.asarray(x)
    total = x.size
    fill = np.sum(x == fill_val)
    ratio = fill / total if total > 0 else 0.0
    logging.info(f"[FILL RATIO WINDOWED] use_train={use_train}, x_shape={x.shape}, fill_val={fill_val}, ratio={ratio:.2%}")

    # 按 channel 统计
    if x.ndim == 4:  # [samples, window, feat, ch]
        ch_cnt = x.shape[-1]
        for ch in range(ch_cnt):
            sub = x[..., ch]
            r = np.mean(sub == fill_val)
            logging.info(f"  channel[{ch}] fill_ratio={r:.2%}")
    return ratio


def inspect_window_monotonic(ds, i=0, use_train=True):
    ch_list = ds.channel_train_x_list if use_train else ds.channel_test_x_list
    main = ch_list[0]
    W = ds.window_size

    win_dates = main[i:i+W, 0].astype(int)
    diffs = np.diff(win_dates)

    print("window idx:", i)
    print("t0:", win_dates[0], "t_last:", win_dates[-1])
    print("date diffs sign (first 10):", np.sign(diffs[:10]))
    print("is strictly decreasing:", np.all(diffs < 0))
    print("is strictly increasing:", np.all(diffs > 0))
    print("head dates:", win_dates[:3])
    print("tail dates:", win_dates[-3:])

def inspect_window_monotonic_effective(ds, sample_idx=0, use_train=True):
    """
    验证“模型真实输入窗口”的时间顺序，而不是原始数据顺序。
    原始数据始终是新->旧；如果启用了 reverse_time_to_ascending=True，
    那么模型窗口应该是旧->新（即 raw_window[::-1]）。
    """
    W = ds.window_size

    # 取主通道原始日期序列（训练/测试）
    if ds.use_conv2_channel:
        ch_list = ds.channel_train_x_list if use_train else ds.channel_test_x_list
        main = ch_list[0]  # shape [N, cols]，第0列是日期
        dates = main[:, 0].astype(int)
    else:
        # 单通道情况下，用 ds.raw_data[:,0]
        dates = ds.raw_data[:, 0].astype(int)

    raw_window_dates = dates[sample_idx: sample_idx + W]
    if raw_window_dates.shape[0] < W:
        raise ValueError(f"sample_idx={sample_idx} window exceeds data length")

    # 你现在 get_windowed_x_by_raw 默认会 reverse_time_to_ascending=True
    model_window_dates = raw_window_dates[::-1]

    print("=== RAW window dates (what you currently printed) ===")
    print("t0(raw head):", raw_window_dates[0], "t_last(raw tail):", raw_window_dates[-1])
    diffs_raw = np.diff(raw_window_dates)
    print("raw strictly decreasing:", np.all(diffs_raw < 0))
    print("raw strictly increasing:", np.all(diffs_raw > 0))
    print("raw head:", raw_window_dates[:3])
    print("raw tail:", raw_window_dates[-3:])

    print("\n=== MODEL window dates (after reverse_time_to_ascending) ===")
    print("t_old(model head):", model_window_dates[0], "t_new(model tail):", model_window_dates[-1])
    diffs_model = np.diff(model_window_dates)
    print("model strictly decreasing:", np.all(diffs_model < 0))
    print("model strictly increasing:", np.all(diffs_model > 0))
    print("model head:", model_window_dates[:3])
    print("model tail:", model_window_dates[-3:])