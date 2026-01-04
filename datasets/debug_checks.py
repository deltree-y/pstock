# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
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

    # =========================
# Single-channel sanity checks (added 2026-01-03)
# =========================
import numpy as np


def _require_single_channel(ds):
    if getattr(ds, "use_conv2_channel", False):
        raise ValueError("This sanity check only supports single-channel: set use_conv2_channel=False.")


def _as_date_val_like(arr_date_col, date_str_or_int):
    """Cast date into same dtype as arr_date_col elements."""
    return type(arr_date_col[0])(date_str_or_int)


def check_predictable_dataset_shape_single(ds, date, verbose=True):
    """
    单通道：仅检查 get_predictable_dataset_by_date(date) 输出形状是否符合约定：
      returns (raw_x, x, close)
      raw_x: [window, feat]
      x:     [1, window, feat]
    """
    _require_single_channel(ds)
    raw_x, x, close = ds.get_predictable_dataset_by_date(date)

    if verbose:
        print("=" * 90)
        print(f"[SanityCheck][Single] check_predictable_dataset_shape_single(date={date})")
        print(f"raw_x.shape={getattr(raw_x, 'shape', None)}, x.shape={getattr(x, 'shape', None)}, close={float(close):.3f}")

    assert raw_x.ndim == 2, f"raw_x must be 2D [window, feat], got ndim={raw_x.ndim}"
    assert x.ndim == 3, f"x must be 3D [1, window, feat], got ndim={x.ndim}"
    assert raw_x.shape[0] == ds.window_size, f"raw_x rows must be window_size={ds.window_size}, got {raw_x.shape[0]}"
    assert x.shape[0] == 1 and x.shape[1] == ds.window_size, f"x must be (1, {ds.window_size}, feat), got {x.shape}"

    if verbose:
        print("[OK] shape check passed.")
        print("=" * 90)


def check_predictable_dataset_alignment_single(ds, date, check_t2=True, verbose=True):
    """
    单通道强校验（deterministic）：
    1) 验证 get_predictable_dataset_by_date(date) 返回的 close 是 T0 close
    2) 验证 raw_x[-1, close_col] == close
    3) 严格复算 ds.raw_y 对应 date 的 t1l/t1h/t2l/t2h 是否与 raw_data 一致（off-by-one 检测核心）

    依赖：
    - ds.raw_data 第0列=日期, 后续列=特征
    - ds.datasets_date_list 与 ds.raw_y 对齐（datasets是 combine_data_np，含y；raw_y来自datasets最后4列）
    - Trade.update_t1_change_rate / update_t2_change_rate 的定义（升序 old->new）
    """
    _require_single_channel(ds)

    # ---- 1) 调用接口，获取窗口与 close ----
    raw_x, x, close = ds.get_predictable_dataset_by_date(date)
    close = float(close)

    # ---- 2) 定位 T0 在 raw_data 中的 idx ----
    # 注意：get_predictable_dataset_by_date 内部会先做 si.get_next_or_current_trade_date
    t0 = ds.si.get_next_or_current_trade_date(date)
    t0_val = _as_date_val_like(ds.raw_data[:, 0], t0)

    idx_arr = np.where(ds.raw_data[:, 0] == t0_val)[0]
    assert idx_arr.size > 0, f"T0 {t0} not found in ds.raw_data"
    i0 = int(idx_arr[0])

    # ---- 3) 校验 close 一致性 ----
    col_close_raw_data = ds.p_trade.col_close + 1  # raw_data 第0列是日期，所以 +1
    close_from_raw_data = float(ds.raw_data[i0, col_close_raw_data])

    # raw_x 不含日期列，所以 close 在 raw_x 内的列索引是 ds.p_trade.col_close
    close_from_raw_x = float(raw_x[-1, ds.p_trade.col_close])

    if verbose:
        print("=" * 90)
        print(f"[SanityCheck][Single] check_predictable_dataset_alignment_single(date={date} -> T0={t0})")
        print(f"T0 idx in raw_data: {i0}")
        print(f"close (returned)      : {close:.6f}")
        print(f"close (raw_data[i0])  : {close_from_raw_data:.6f}  (col={col_close_raw_data})")
        print(f"close (raw_x[-1])     : {close_from_raw_x:.6f}  (col={ds.p_trade.col_close})")
        print(f"raw_x.shape={raw_x.shape}, x.shape={x.shape}")

    assert abs(close - close_from_raw_data) < 1e-8, "Returned close != raw_data T0 close (possible wrong index/+1 bug)"
    assert abs(close - close_from_raw_x) < 1e-8, "Returned close != raw_x last-step close (window last day not T0?)"

    # ---- 4) 复算 t1/t2 change rate，并对比 ds.raw_y 的同日标签 ----
    # ds.raw_y 是从 datasets(combine_data_np) 抽出最后4列，且 datasets_date_list 是 datasets 的日期列
    # 因此用 datasets_date_list 定位 raw_y 行
    t0_val_ds = _as_date_val_like(ds.datasets_date_list, t0)
    idxy_arr = np.where(ds.datasets_date_list == t0_val_ds)[0]
    assert idxy_arr.size > 0, f"T0 {t0} not found in ds.datasets_date_list (cannot locate ds.raw_y row)"
    iy = int(idxy_arr[0])

    y_ds = ds.raw_y[iy].astype(float)  # [t1l, t1h, t2l, t2h] (small decimals)
    ds_t1l, ds_t1h, ds_t2l, ds_t2h = map(float, y_ds.tolist())

    # raw_data: i1=i0+1, i2=i0+2
    i1 = i0 + 1
    assert i1 < ds.raw_data.shape[0], f"Not enough data for T1 at T0={t0} (i0={i0})"

    col_low_raw_data = ds.p_trade.col_low + 1
    col_high_raw_data = ds.p_trade.col_high + 1
    t1_low = float(ds.raw_data[i1, col_low_raw_data])
    t1_high = float(ds.raw_data[i1, col_high_raw_data])

    calc_t1l = (t1_low - close_from_raw_data) / close_from_raw_data
    calc_t1h = (t1_high - close_from_raw_data) / close_from_raw_data

    if verbose:
        print("-" * 90)
        print(f"ds.raw_y(T0)    : t1l={ds_t1l:.10f}, t1h={ds_t1h:.10f}, t2l={ds_t2l:.10f}, t2h={ds_t2h:.10f}")
        print(f"calc from prices: t1l={calc_t1l:.10f}, t1h={calc_t1h:.10f}")

    assert abs(ds_t1l - calc_t1l) < 1e-8, "T1L mismatch (likely off-by-one in label alignment)"
    assert abs(ds_t1h - calc_t1h) < 1e-8, "T1H mismatch (likely off-by-one in label alignment)"

    if check_t2:
        i2 = i0 + 2
        assert i2 < ds.raw_data.shape[0], f"Not enough data for T2 at T0={t0} (i0={i0})"
        t2_low = float(ds.raw_data[i2, col_low_raw_data])
        t2_high = float(ds.raw_data[i2, col_high_raw_data])
        calc_t2l = (t2_low - close_from_raw_data) / close_from_raw_data
        calc_t2h = (t2_high - close_from_raw_data) / close_from_raw_data

        if verbose:
            print(f"calc from prices: t2l={calc_t2l:.10f}, t2h={calc_t2h:.10f}")

        assert abs(ds_t2l - calc_t2l) < 1e-8, "T2L mismatch (likely off-by-one in label alignment)"
        assert abs(ds_t2h - calc_t2h) < 1e-8, "T2H mismatch (likely off-by-one in label alignment)"

    if verbose:
        print("[OK] alignment check passed (T0 close + T1/T2 rates).")
        print("=" * 90)


def check_train_window_basic_single(ds, n_samples=10, seed=2025, use_train=True):
    """
    单通道弱校验（抽样）：
    - 检查 windowed X 与 y 是否存在 NaN/inf
    - 打印若干样本的 last close、raw_y、binned/binary y（便于肉眼看有没有明显异常）
    不做“日期精确对齐”验证（因为训练窗口不带日期列）。
    """
    _require_single_channel(ds)
    rng = np.random.default_rng(seed)

    wx = ds.raw_windowed_train_x if use_train else ds.raw_windowed_test_x
    y = ds.train_y if use_train else ds.test_y
    raw_y = ds.raw_train_y if use_train else ds.raw_test_y
    tag = "TRAIN" if use_train else "TEST"

    if wx is None or y is None or raw_y is None:
        raise ValueError(f"{tag}: data is None (note: test_x only exists when train_size < 1)")

    n = wx.shape[0]
    if n == 0:
        raise ValueError(f"{tag}: empty windowed x")

    pick = rng.choice(n, size=min(n_samples, n), replace=False)

    # NaN/inf quick scan
    nan_cnt = int(np.isnan(wx).sum()) + int(np.isnan(y).sum()) + int(np.isnan(raw_y).sum())
    inf_cnt = int(np.isinf(wx).sum()) + int(np.isinf(y).sum()) + int(np.isinf(raw_y).sum())
    print("=" * 90)
    print(f"[SanityCheck][Single][{tag}] check_train_window_basic_single samples={len(pick)}, window={ds.window_size}")
    print(f"NaN count total={nan_cnt}, Inf count total={inf_cnt}")
    print("-" * 90)

    for i in pick:
        last_close = float(wx[i, -1, ds.p_trade.col_close])
        print(f"[{tag}] idx={i:6d} last_close={last_close:.3f} y={y[i]} raw_y(t1l,t1h,t2l,t2h)={raw_y[i]}")

    print("=" * 90)