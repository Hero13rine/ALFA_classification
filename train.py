import numpy as np
import tensorflow as tf
import glob
import os

from config import *
from models.lstm_model import build_lstm_model
from utils.data_loader import (
    compute_mean_std_from_labeled_normals,
    RNN_set_making,
    split_per_class
)
from utils.metrics import plot_training_history, evaluate_model
from callbacks.per_class_callback import PerClassAccuracyCallback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# === 加载数据 ===
data_files = sorted(glob.glob("data/*.csv"))
print(f"Found {len(data_files)} CSV files for training.")

# === 计算标准化参数 ===
normal_mean, normal_std = compute_mean_std_from_labeled_normals(
    data_files, REMOVE_STEP, STATE_INPUT_DIM)

# === 构建样本集 ===
X, y, _, _ = RNN_set_making(
    data_files, WINDOW_SIZE, normal_mean, normal_std, STATE_INPUT_DIM, REMOVE_STEP)

# === 划分数据集 ===
X_train, y_train, X_val, y_val, X_test, y_test = split_per_class(
    X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
    min_samples=MIN_SAMPLES_PER_CLASS)

# === 模型构建 ===
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape, num_classes=NUM_CLASSES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 设置回调 ===
per_class_cb = PerClassAccuracyCallback(X_val, y_val, label_map=LABEL_MAP)
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=20, min_delta=1e-4,
    restore_best_weights=True, verbose=1)
red_cb = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
model_cb = ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),

# === 模型训练 ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[per_class_cb,
               early_stop_cb,
               red_cb,
               model_cb],
    # class_weight=
)

# === 保存模型与训练历史 ===
os.makedirs("outputs", exist_ok=True)
model.save("outputs/model.h5")
plot_training_history(history)

# === 模型评估与可视化报告 ===
evaluate_model(model, X_test, y_test, label_map=LABEL_MAP)