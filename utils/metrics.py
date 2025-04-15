import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def plot_training_history(history, save_path="outputs/training_history.png"):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, X_test, y_test, label_map, output_dir="outputs"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 获取预测结果
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # 分类报告
    report_str = classification_report(y_true_classes, y_pred_classes, target_names=list(label_map.values()))
    print("Classification Report:\n")
    print(report_str)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report_str)

    # 混淆矩阵
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=list(label_map.values()))
    disp_norm.plot(cmap=plt.cm.Greens)
    plt.title("Normalized Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix_normalized.png")
    plt.close()

    # 每类准确率输出
    acc_lines = []
    acc_lines.append("Per-class test accuracy:\n")
    for i in range(len(label_map)):
        idx = np.where(y_true_classes == i)[0]
        acc = accuracy_score(y_true_classes[idx], y_pred_classes[idx]) if len(idx) > 0 else 0
        acc_lines.append(f"  {label_map[i]}: {acc:.4f}")
    print("\n".join(acc_lines))

    with open(f"{output_dir}/per_class_test_accuracy.txt", "w") as f:
        f.write("\n".join(acc_lines))