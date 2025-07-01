#!/usr/bin/env python3
"""
run_inference_v2.py  ——  推理 + 双指标（首次告警 & 有效告警）评估
-----------------------------------------------------------------
* 记录 2 个告警步序：
    1. first_alarm_idx   —— 首次满足阈值且连续 stable_steps 的告警
    2. valid_alarm_idx   —— 位于 inj_idx‑LEAD_STEPS 之后的首次告警（排除早期误报）
* 分别计算 TTD/ARL，并统计误报次数。

用法：
    python run_inference_v2.py  # 与 run_inference.py 相同
"""
import os, glob, json
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ===== 项目内工具 =====
from utils.data_loader import (
    data_preprocessing,
    compute_mean_std_from_labeled_normals,
)
from utils.early_warning_utils import (
    stable_alarm_steps,
    preprocess_window,
)
from utils.metrics import compute_ttd_arl  # 原版函数即可

# ===== 配置：与训练保持一致 =====
from config import (
    STATE_INPUT_DIM,
    WINDOW_SIZE,
    LEAD_STEPS,
    LABEL_MAP,
)

DATA_DIR        = "data/fault"
MODEL_PATH      = "outputs/weighted_ce/model.h5"
OUT_DIR         = "inference_results_v2"

FAULT_CLASS     = 1          # 发动机故障索引
ALARM_THRESH    = 0.7
STABLE_STEPS    = 3
SAMPLE_RATE     = 4.0        # 4 Hz

# -------------------------------------------------------------
# 辅助：单条 CSV → (data, labels, inj_idx)
# -------------------------------------------------------------

def load_sample(fp: str):
    raw = np.genfromtxt(fp, delimiter=",")
    seq, labels_onehot = data_preprocessing(raw, 0, STATE_INPUT_DIM)
    # 故障注入步序：第一次出现 fault_class==1
    mask = labels_onehot[:, FAULT_CLASS] == 1
    inj_idx = int(np.argmax(mask)) if mask.any() else None
    return seq[:, :STATE_INPUT_DIM], labels_onehot, inj_idx

# -------------------------------------------------------------
# 滑窗推理（与训练窗口对齐，1‑step stride）
# -------------------------------------------------------------

def sliding_predict(model, seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    T = seq.shape[0]
    n_cls = model.output_shape[-1]
    probs = np.zeros((T, n_cls), dtype=np.float32)
    for end in range(WINDOW_SIZE, T + 1):
        win = preprocess_window(seq[end - WINDOW_SIZE : end], mean, std)
        win = win[None, ...]
        probs[end - 1] = model.predict(win, verbose=0)[0]
    return probs

# -------------------------------------------------------------
# 主流程
# -------------------------------------------------------------

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    assert files, f"No CSV found in {DATA_DIR}"

    # mean/std 取训练相同策略
    mean_vec, std_vec = compute_mean_std_from_labeled_normals(
        files, remove_step=0, state_input_dim=STATE_INPUT_DIM
    )

    # 加载模型
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    inj_steps: List[Optional[int]] = []
    first_alarm_steps: List[Optional[int]] = []
    valid_alarm_steps: List[Optional[int]] = []
    seq_lens: List[int] = []
    flight_log = []

    for fp in tqdm(files, desc="Inference"):
        data, _, inj_idx = load_sample(fp)
        seq_lens.append(len(data))

        probs = sliding_predict(model, data, mean_vec, std_vec)
        first_alarm_idx = stable_alarm_steps(
            probs, ALARM_THRESH, STABLE_STEPS, FAULT_CLASS
        )

        # === 剔除 inj_idx‑LEAD_STEPS 之前的误报 ===
        if inj_idx is not None and first_alarm_idx is not None and first_alarm_idx < inj_idx - LEAD_STEPS:
            valid_alarm_idx = None  # 提前误报 → 无有效告警
        else:
            valid_alarm_idx = first_alarm_idx

        inj_steps.append(inj_idx)
        first_alarm_steps.append(first_alarm_idx)
        valid_alarm_steps.append(valid_alarm_idx)

        flight_log.append(
            {
                "file": Path(fp).name,
                "inj_idx": inj_idx,
                "first_alarm_idx": first_alarm_idx,
                "valid_alarm_idx": valid_alarm_idx,
                "first_alarm_time_s": None if first_alarm_idx is None else first_alarm_idx / SAMPLE_RATE,
                "valid_alarm_time_s": None if valid_alarm_idx is None else valid_alarm_idx / SAMPLE_RATE,
                "first_alarm_prob": None if first_alarm_idx is None else float(probs[first_alarm_idx, FAULT_CLASS]),
                "valid_alarm_prob": None if valid_alarm_idx is None else float(probs[valid_alarm_idx, FAULT_CLASS]) if valid_alarm_idx is not None else None,
            }
        )

    # ===== 统计指标 =====
    metrics_first  = compute_ttd_arl(inj_steps, first_alarm_steps, SAMPLE_RATE)
    metrics_valid  = compute_ttd_arl(inj_steps, valid_alarm_steps, SAMPLE_RATE)

    # 误报统计（提前误报次数 / 占比）
    num_fp = sum(
        1
        for inj, alarm in zip(inj_steps, first_alarm_steps)
        if alarm is not None and (inj is None or alarm < inj - LEAD_STEPS-5)
    )
    fp_rate = num_fp / len(files)

    # ===== 保存 =====
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUT_DIR}/flight_alarm_log.json", "w") as f:
        json.dump(flight_log, f, indent=2)
    with open(f"{OUT_DIR}/metrics_first_alarm.json", "w") as f:
        json.dump(metrics_first, f, indent=2)
    with open(f"{OUT_DIR}/metrics_valid_alarm.json", "w") as f:
        json.dump(metrics_valid, f, indent=2)
    with open(f"{OUT_DIR}/false_positive_stats.json", "w") as f:
        json.dump({"false_positive_count": num_fp, "false_positive_rate": fp_rate}, f, indent=2)

    # ===== 控制台输出 =====
    print("\n==== Early‑Warning Summary (First Alarm) ====")
    for k, v in metrics_first.items():
        print(f"{k}: {v:.3f}")
    print("\n==== Early‑Warning Summary (Valid Alarm) ====")
    for k, v in metrics_valid.items():
        print(f"{k}: {v:.3f}")
    print(f"\nFalse‑positive count: {num_fp}  (rate = {fp_rate:.2%})")
    print(f"Results saved to → {OUT_DIR}")


if __name__ == "__main__":
    main()
