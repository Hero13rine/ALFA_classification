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

def get_loss_function(loss_type, y_train=None):
    if loss_type == 'cross_entropy':
        return 'categorical_crossentropy'
    elif loss_type == 'weighted_cross_entropy':
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        import tensorflow.keras.backend as K

        y_int = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        def weighted_loss(y_true, y_pred):
            weights = tf.reduce_sum(class_weights_tensor * y_true, axis=-1)
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return loss * weights

        return weighted_loss

    elif loss_type == 'focal_loss':
        def focal_loss(gamma=2., alpha=0.25):
            def focal_loss_fixed(y_true, y_pred):
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
                cross_entropy = -y_true * tf.math.log(y_pred)
                loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
                return tf.reduce_sum(loss, axis=-1)
            return focal_loss_fixed
        return focal_loss()


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
    loss_fn = get_loss_function(config.get("loss_type", "cross_entropy"), y_train)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

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