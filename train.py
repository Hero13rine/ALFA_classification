# train.py  —— 训练 + 分类评估 + 早期预警评估
# ------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from pathlib import Path
import glob, os
from tqdm import tqdm

from models.lstm_model import build_lstm_model
from utils.data_loader import (
    compute_mean_std_from_labeled_normals,
    RNN_set_making,
    split_per_class
)
from utils.metrics import plot_training_history, evaluate_model
from utils.early_warning_utils import (
    load_alfa_sample, sliding_predict, stable_alarm_steps
)
# 若你用不到，可注释掉下行
from callbacks.per_class_callback import PerClassAccuracyCallback


# ------------------------------------------------------------
# 1) LOSS 选择器（与你原来一致）
# ------------------------------------------------------------
def get_loss_function(loss_type: str, y_train=None):
    if loss_type == "cross_entropy":
        return "categorical_crossentropy"

    if loss_type == "weighted_cross_entropy":
        from sklearn.utils.class_weight import compute_class_weight
        cls_w = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(np.argmax(y_train, 1)),
            y=np.argmax(y_train, 1)
        )
        cls_w = tf.constant(cls_w, dtype=tf.float32)

        def weighted(y_true, y_pred):
            w = tf.reduce_sum(cls_w * y_true, axis=-1)
            base = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return base * w
        return weighted

    if loss_type == "focal_loss":
        γ, α = 2.0, 0.25
        def focal(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
            ce = -y_true * tf.math.log(y_pred)
            return tf.reduce_sum(α * tf.pow(1 - y_pred, γ) * ce, axis=-1)
        return focal


# ------------------------------------------------------------
# 2) 主函数：接收字典 config，负责一次完整实验
# ------------------------------------------------------------
def train_with_params(config: dict):
    print(f"\n[INFO] ▶▶ 运行实验: {config.get('tag', 'unnamed')}")

    # ---------- 路径 ----------
    data_files = sorted(glob.glob(os.path.join(config["data_dir"], "*.csv")))
    if not data_files:
        raise RuntimeError(f"在 {config['data_dir']} 没找到 CSV!")

    # ---------- 标准化 ----------
    mean_vec, std_vec = compute_mean_std_from_labeled_normals(
        data_files,
        config["remove_step"],
        config["state_input_dim"]
    )

    # ---------- 滑窗采样 (输入 → 未来 lead_steps 标签) ----------
    X, y, *_ = RNN_set_making(
        data_files,
        window       = config["window_size"],
        normal_mean  = mean_vec,
        normal_std   = std_vec,
        state_input_dim = config["state_input_dim"],
        remove_step  = config["remove_step"],
        lead_steps   = config["lead_steps"],
        ignore_tail  = True
    )

    # ---------- 数据划分 ----------
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_per_class(
        X, y,
        train_ratio = config["train_ratio"],
        val_ratio   = config["val_ratio"],
        test_ratio  = config["test_ratio"],
        min_samples = config["min_samples_per_class"]
    )

    # ---------- 构建 & 编译模型 ----------
    model = build_lstm_model(
        input_shape = (X_tr.shape[1], X_tr.shape[2]),
        num_classes = config["num_classes"]
    )
    model.compile(
        optimizer = "adam",
        loss      = get_loss_function(config["loss_type"], y_tr),
        metrics   = ["accuracy"]
    )

    cbks = [
        EarlyStopping(monitor="val_accuracy", patience=20,
                      min_delta=1e-4, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        ModelCheckpoint("best_model.h5", save_best_only=True)
    ]
    # 若有 callback 源码则保留
    if "PerClassAccuracyCallback" in globals():
        cbks.insert(0, PerClassAccuracyCallback(
            X_val, y_val, label_map=config["label_map"])
        )

    history = model.fit(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs          = config["epochs"],
        batch_size      = config["batch_size"],
        callbacks       = cbks,
        verbose         = 2
    )

    # ---------- 输出目录 ----------
    out_dir = Path(f"outputs/{config.get('tag','exp')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.h5")
    plot_training_history(history, out_dir / "training_history.png")

    # ---------- 分类评估 ----------
    evaluate_model(
        model,
        X_te, y_te,
        label_map = config["label_map"],
        output_dir= str(out_dir)
    )


# ------------------------------------------------------------
# 3) CLI（由 run_experiment.py 调用）
# ------------------------------------------------------------
if __name__ == "__main__":
    # 直接跑单实验时：
    #   python train.py
    # 会加载 config.py 里的 DEFAULT_CONFIG
    import config as cfg
    train_with_params(cfg.DEFAULT_CONFIG)
