import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import os
import pandas as pd
from typing import List, Optional, Tuple, Dict


# ------------------------------------------------------------------
# ★★★ Early-warning metrics ★★★
# ------------------------------------------------------------------
def compute_ttd_arl(
    inj_steps: List[int],
    alarm_steps: List[Optional[int]],
    sample_rate: float = 4.0,
) -> Dict[str, float]:
    """
    计算提前量 (TTD) 与平均运行长度 (ARL)。

    Parameters
    ----------
    inj_steps   : list[int]
        每条样本真实故障注入在第几步（0-idx）——长度 = N。
    alarm_steps : list[int or None]
        每条样本第一次稳定告警触发的步序号；None 表示无告警。
    sample_rate : float
        采样率 Hz（4Hz ⇒ 1 步 = 0.25s）。

    Returns
    -------
    dict  { "mean_ttd": …, "std_ttd": …,
            "mean_arl": …, "std_arl": … }
        - TTD = (inj_step − alarm_step) / sample_rate  (秒；正值=提前)
        - ARL 用正常段长度 / sample_rate  近似估计
    """
    assert len(inj_steps) == len(
        alarm_steps
    ), "inj_steps and alarm_steps length mismatch"
    ttd_list = []
    arl_list = []

    for inj, alarm in zip(inj_steps, alarm_steps):
        # ---- TTD 只在成功预警时统计 ----
        if alarm is not None and alarm < inj:
            ttd_list.append((inj - alarm) / sample_rate)
        # ---- ARL：统计故障注入前是否已误报 ----
        if alarm is None:
            arl_list.append(inj / sample_rate)  # 无误报 → 至少运行到 inj
        elif alarm < inj:
            arl_list.append(alarm / sample_rate)  # 误报时间
        else:
            # alarm >= inj 视作未误报
            arl_list.append(inj / sample_rate)

    def _avg_std(arr):
        return (float(np.mean(arr)), float(np.std(arr))) if arr else (np.nan, np.nan)

    mean_ttd, std_ttd = _avg_std(ttd_list)
    mean_arl, std_arl = _avg_std(arl_list)

    return dict(
        mean_ttd=mean_ttd,
        std_ttd=std_ttd,
        mean_arl=mean_arl,
        std_arl=std_arl,
    )


# ------------------------------------------------------------------
# 原有可视化/评估函数
# ------------------------------------------------------------------
def plot_training_history(history, save_path: str = "outputs/training_history.png"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_acc, "ro-", label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training Loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: Dict[int, str],
    output_dir: str = "outputs",
    inj_steps: Optional[List[int]] = None,
    alarm_steps: Optional[List[Optional[int]]] = None,
):
    """
    通用评估 + 可选早期预警指标（TTD / ARL）。

    Parameters
    ----------
    model       : 已训练好的 Keras / PyTorch 模型 (需有 .predict)
    X_test, y_test : 测试集
    label_map   : {idx: name}
    output_dir  : 结果保存目录
    inj_steps   : list[int]  (可选) 真实注入步序
    alarm_steps : list[int or None]  (可选) 模型首次稳定告警步序
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- 分类评估 ----------------
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    report_str = classification_report(
        y_true_classes, y_pred_classes, target_names=list(label_map.values())
    )
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report_str)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(label_map.values())
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=list(label_map.values())
    )
    disp_norm.plot(cmap=plt.cm.Greens)
    plt.title("Normalized Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix_normalized.png")
    plt.close()

    acc_lines = ["Per-class test accuracy:\n"]
    for i in range(len(label_map)):
        idx = np.where(y_true_classes == i)[0]
        acc = (
            accuracy_score(y_true_classes[idx], y_pred_classes[idx])
            if len(idx) > 0
            else 0
        )
        acc_lines.append(f"  {label_map[i]}: {acc:.4f}")
    with open(f"{output_dir}/per_class_test_accuracy.txt", "w") as f:
        f.write("\n".join(acc_lines))

    # ---- 四格统计 ----
    total = np.sum(cm)
    num_classes = len(label_map)
    four_metrics = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total - TP - FP - FN
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        acc = (TP + TN) / total if total > 0 else 0
        four_metrics.append(
            [
                label_map[i],
                TP,
                FP,
                FN,
                TN,
                round(precision, 4),
                round(recall, 4),
                round(f1, 4),
                round(acc, 4),
            ]
        )

    df_metrics = pd.DataFrame(
        four_metrics,
        columns=[
            "Class",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1-score",
            "Accuracy",
        ],
    )
    df_metrics.to_csv(f"{output_dir}/per_class_metrics_table.csv", index=False)
    print(
        "\nPer-Class Metrics Table written to:",
        f"{output_dir}/per_class_metrics_table.csv",
    )

    # ---------------- 提前量 & ARL ----------------
    if inj_steps is not None and alarm_steps is not None:
        early_metrics = compute_ttd_arl(inj_steps, alarm_steps, sample_rate=4.0)
        with open(f"{output_dir}/early_warning_metrics.txt", "w") as f:
            f.write(
                "Early-Warning Metrics\n"
                "---------------------\n"
                f"Average TTD (s): {early_metrics['mean_ttd']:.3f} "
                f"± {early_metrics['std_ttd']:.3f}\n"
                f"Average ARL (s): {early_metrics['mean_arl']:.3f} "
                f"± {early_metrics['std_arl']:.3f}\n"
            )
        print(
            "Early-Warning metrics written to:",
            f"{output_dir}/early_warning_metrics.txt",
        )
