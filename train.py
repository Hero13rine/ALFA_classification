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


def train_with_params(config):
    print(f"Running experiment with config: {config}")

    # === 加载数据 ===
    data_files = sorted(glob.glob("data/*.csv"))

    # === 计算标准化参数 ===
    normal_mean, normal_std = compute_mean_std_from_labeled_normals(
        data_files, config["remove_step"], config["state_input_dim"])

    # === 构建样本集 ===
    X, y, _, _ = RNN_set_making(
        data_files, config["window_size"], normal_mean, normal_std,
        config["state_input_dim"], config["remove_step"])

    # === 划分数据集 ===
    X_train, y_train, X_val, y_val, X_test, y_test = split_per_class(
        X, y, train_ratio=config["train_ratio"], val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"], min_samples=config["min_samples_per_class"])

    # === 模型构建 ===
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, num_classes=config["num_classes"])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # === 设置回调 ===
    per_class_cb = PerClassAccuracyCallback(X_val, y_val, label_map=config["label_map"])
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=20, min_delta=1e-4,
        restore_best_weights=True, verbose=0)
    red_cb = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    model_cb = ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),


    # === 模型训练 ===
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[per_class_cb,
                   early_stop_cb,
                   red_cb,
                   model_cb],
        # class_weight=
    )

    # === 输出目录命名 ===
    tag = config.get("tag", "default")
    outdir = f"outputs/experiment_{tag}"
    os.makedirs(outdir, exist_ok=True)

    # === 保存模型与评估 ===
    model.save(f"{outdir}/model.h5")
    plot_training_history(history, save_path=f"{outdir}/training_history.png")
    evaluate_model(model, X_test, y_test, label_map=config["label_map"], output_dir=outdir)

    return history