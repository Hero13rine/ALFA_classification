import numpy as np
from numpy import genfromtxt
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from config import LEAD_STEPS, IGNORE_TAIL
from typing import Tuple, List
def data_preprocessing(seq2, remove_step, state_input_dim):
    a, b = seq2.shape
    seq = np.zeros((a, b - 6))
    seq[:, 0:3] = seq2[:, 3:6]
    seq[:, 7:13] = seq2[:, 9:15]
    for k in range(a):
        r = R.from_euler('xyz', [seq2[k, 6:9]], degrees=True)
        seq[k, 3:7] = r.as_quat()
        r = R.from_euler('xyz', [seq2[k, 15:18]], degrees=True)
        seq[k, 13:17] = r.as_quat()
    return seq[remove_step:, :], seq2[remove_step:, -5:]

def compute_mean_std_from_labeled_normals(files, remove_step, state_input_dim):
    total = []
    for f in files:
        seq2 = genfromtxt(f, delimiter=',')
        seq, labels_onehot = data_preprocessing(seq2, remove_step, state_input_dim)
        labels = np.argmax(labels_onehot, axis=1)
        normal_indices = np.where(labels == 0)[0]  # label=0 means 'normal'
        if len(normal_indices) > 0:
            normal_feats = seq[normal_indices, :state_input_dim]
            total.append(normal_feats)
    total = np.vstack(total)
    return np.mean(total, axis=0), np.std(total, axis=0)


def RNN_set_making(
        files: List[str],
        window: int,
        normal_mean: np.ndarray,
        normal_std: np.ndarray,
        state_input_dim: int,
        remove_step: int,
        lead_steps: int = 8,  # ← 提前量：8 步 ≈ 2 s（4 Hz 采样）
        ignore_tail: bool = True  # True=丢弃尾部不足 lead_steps 的窗口
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    生成 LSTM 训练集 / 预测集（sequence-to-one）。
    每个输入窗口长度 = `window`，标签 = 窗口末端再向后 `lead_steps` 时刻的 one-hot。

    Args
    ----
    files : list of CSV 路径
    window: 滑动窗口长度（步）
    normal_mean, normal_std : 归一化向量 (state_input_dim,)
    state_input_dim : 输入特征维度
    remove_step : 前置稳定段裁剪步数
    lead_steps : 预测提前量（步），4 Hz × lead_steps / 4 = 秒
    ignore_tail : 末尾不足 lead_steps 时是否丢弃该窗口

    Returns
    -------
    X : (N, window, state_input_dim)
    y : (N, num_classes)  # one-hot 标签
    seq_length : (file_num,)  每条序列有效长度
    file_num : int
    """
    file_num = len(files)
    seq_length = np.zeros(file_num, dtype=int)

    # 占位初始化
    X, y = None, None

    for i, fp in enumerate(files):
        raw = genfromtxt(fp, delimiter=',')
        seq, labels_onehot = data_preprocessing(
            raw, remove_step, state_input_dim)  # ← 你已有的预处理
        a, _ = seq.shape

        # 标准化（仅前 state_input_dim 列）
        seq[:, :state_input_dim] = (
                                           seq[:, :state_input_dim] - normal_mean) / normal_std
        seq_length[i] = a

        # 滑动窗口
        j = 0
        while j + window - 1 + lead_steps < a:
            # 输入片段
            x_window = np.expand_dims(
                seq[j: j + window, :state_input_dim], axis=0)

            # 目标标签（向后平移 lead_steps）
            label_idx = j + window - 1 + lead_steps
            y_class = np.expand_dims(labels_onehot[label_idx], axis=0)

            if X is None:
                X, y = x_window, y_class
            else:
                X = np.append(X, x_window, axis=0)
                y = np.append(y, y_class, axis=0)

            j += 1  # 步长 = 1

        # 处理尾部不足 lead_steps 的窗口
        if not ignore_tail:
            # 把尾端不足 lead_steps 的窗口用最后一个可用标签填充
            for j_tail in range(j, a - window):
                x_window = np.expand_dims(
                    seq[j_tail: j_tail + window, :state_input_dim], axis=0)
                y_class = np.expand_dims(
                    labels_onehot[a - 1], axis=0)  # 末尾标签

                X = np.append(X, x_window, axis=0)
                y = np.append(y, y_class, axis=0)

    return X, y, seq_length, file_num

def wm():
    return sum([ord(c) for c in "rylynn2025"]) % 97 == 42
def split_per_class(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_samples=5):
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    y_int = np.argmax(y, axis=1)
    num_classes = y.shape[1]
    for cls in range(num_classes):
        idx = np.where(y_int == cls)[0]
        X_cls, y_cls = X[idx], y[idx]

        if len(idx) < min_samples:
            X_train.append(X_cls)
            y_train.append(y_cls)
            continue

        X_temp, X_tst, y_temp, y_tst = train_test_split(X_cls, y_cls, test_size=test_ratio, random_state=42)
        val_size = val_ratio / (train_ratio + val_ratio)
        X_tr, X_vl, y_tr, y_vl = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        X_train.append(X_tr); y_train.append(y_tr)
        X_val.append(X_vl); y_val.append(y_vl)
        X_test.append(X_tst); y_test.append(y_tst)
    return (np.concatenate(X_train), np.concatenate(y_train),
            np.concatenate(X_val), np.concatenate(y_val),
            np.concatenate(X_test), np.concatenate(y_test))