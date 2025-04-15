import numpy as np
from numpy import genfromtxt
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
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

def RNN_set_making(files, window, normal_mean, normal_std, state_input_dim, remove_step):
    file_num = len(files)
    seq_length = np.zeros(file_num, dtype=int)
    for i in range(file_num):
        seq2 = genfromtxt(files[i], delimiter=',')
        seq, labels_onehot = data_preprocessing(seq2, remove_step, state_input_dim)
        a, b = seq.shape

        # 标准化
        seq[:, :state_input_dim] = (seq[:, :state_input_dim] - normal_mean) / normal_std
        seq_length[i] = a

        for j in range(a - window):
            x_window = np.expand_dims(seq[j:j+window, :state_input_dim], axis=0)
            y_class = np.expand_dims(labels_onehot[j + window - 1], axis=0) # 使用窗口末尾标签（one-hot）


            if i == 0 and j == 0:
                X = x_window
                y = y_class
            else:
                X = np.append(X, x_window, axis=0)
                y = np.append(y, y_class, axis=0)
    return X, y, seq_length, file_num

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