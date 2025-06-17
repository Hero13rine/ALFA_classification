# utils/early_warning_utils.py
# ------------------------------------------------------------
# 一组与训练流程解耦的早期预警辅助函数：
#   1. load_alfa_sample        读取单条 ALFA CSV，返回数据 / 标签 / 注入步
#   2. preprocess_window       按训练同一均值 & 方差做标准化
#   3. sliding_predict         逐步滑窗 → softmax 概率序列
#   4. stable_alarm_steps      连续帧防抖，返回首次“稳定告警”步序
# ------------------------------------------------------------
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

# 如果你的 data_preprocessing() 在别的模块，请按真实路径 import
from utils.data_loader import data_preprocessing


# ============================================================
# 读取单条 CSV → (data, labels_onehot, inj_idx)
# ============================================================
def load_alfa_sample(
    csv_path: str | Path,
    state_input_dim: int,
    fault_class: int = 1,
    remove_step: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Parameters
    ----------
    csv_path         : ALFA 原始 csv 路径
    state_input_dim  : 有效特征维度（前 N 列）
    fault_class      : 要评估的故障类别在 one-hot 中的索引
    remove_step      : 与训练保持一致的“稳定段裁剪”步数

    Returns
    -------
    data             : (T, state_input_dim)   float32
    labels_onehot    : (T, num_classes)       int8
    inj_idx          : int, 第一次出现指定 fault_class =1 的步序 (0-idx)
    """
    arr = np.genfromtxt(csv_path, delimiter=",")
    # 复用你已有的预处理逻辑，保证标签含 one-hot
    data_full, labels_onehot = data_preprocessing(
        arr, remove_step, state_input_dim
    )
    # 裁剪后长度
    T = labels_onehot.shape[0]

    # 故障注入步序：第一次出现指定故障
    fault_mask = labels_onehot[:, fault_class] == 1
    if not fault_mask.any():
        raise ValueError(f"No fault class={fault_class} found in {csv_path}")
    inj_idx = int(np.argmax(fault_mask))  # first True

    # 取有效特征
    data = data_full[:, :state_input_dim].astype(np.float32)

    return data, labels_onehot.astype(np.int8), inj_idx


# ============================================================
# 窗口标准化
# ============================================================
def preprocess_window(
    window: np.ndarray,
    mean_vec: np.ndarray,
    std_vec: np.ndarray,
) -> np.ndarray:
    """
    归一化一个 (window, feat_dim) 的片段，返回相同 shape 的 float32
    """
    return ((window - mean_vec) / std_vec).astype(np.float32)


# ============================================================
# 逐步滑窗预测
# ============================================================
def sliding_predict(
    model,
    seq: np.ndarray,
    mean_vec: np.ndarray,
    std_vec: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Online-like 推理：以滑动窗口输出 softmax 序列

    Parameters
    ----------
    model      : 已加载好权重的 Keras 或 PyTorch 模型
    seq        : (T, feat)  原始特征序列（未归一化）
    mean_vec   : 与训练一致的均值向量
    std_vec    : 与训练一致的方差向量
    window     : 与训练一致的窗口长度

    Returns
    -------
    probs      : (T, num_classes)  每一步 aligned 的 softmax 概率
                 *序列前 window-1 步概率为 0*（因无完整窗口）
    """
    T, feat_dim = seq.shape
    num_classes = model.output_shape[-1]  # Keras；PyTorch 用 .weight.size(0)
    probs = np.zeros((T, num_classes), dtype=np.float32)

    # 逐步滑窗：end_id = window, window+1, …, T
    for end in range(window, T + 1):
        win = seq[end - window : end]
        win = preprocess_window(win, mean_vec, std_vec)
        win = win[None, ...]  # shape (1, window, feat)

        # 这里兼容 Keras / PyTorch 两种 API
        if hasattr(model, "predict"):           # Keras
            p = model.predict(win, verbose=0)[0]
        else:                                   # PyTorch
            import torch
            with torch.no_grad():
                p = model(torch.from_numpy(win).permute(1, 0, 2)).cpu().numpy()[0]
            p = np.exp(p) / np.exp(p).sum()     # softmax

        probs[end - 1] = p

    return probs


# ============================================================
# 连续帧防抖 + 首次稳定告警步序
# ============================================================
def stable_alarm_steps(
    prob_seq: np.ndarray,
    thresh: float,
    stable_steps: int,
    fault_class: int,
) -> Optional[int]:
    """
    Parameters
    ----------
    prob_seq    : (T, num_classes)  softmax 概率时间序列
    thresh      : 置信度阈值
    stable_steps: 连续帧数 N，满足 N 帧 prob>thresh 即算告警
    fault_class : 要监测的故障类别索引

    Returns
    -------
    idx (int) / None
        - 首次稳定告警 **窗口末端** 的步序号 (0-idx)
        - 若序列中从未连续 N 帧超过阈值 ⇒ 返回 None
    """
    above  = prob_seq[:, fault_class] > thresh
    consec = 0
    for t, flag in enumerate(above):
        consec = consec + 1 if flag else 0
        if consec >= stable_steps:
            return t - stable_steps + 1
    return None
