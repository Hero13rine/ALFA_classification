from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import accuracy_score

class PerClassAccuracyCallback(Callback):
    def __init__(self, X_val, y_val, label_map=None):
        self.X_val = X_val
        self.y_val = y_val
        self.label_map = label_map or {}

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_val, axis=1)

        print(f"\nEpoch {epoch+1} - Per-class validation accuracy:")
        for i in range(self.y_val.shape[1]):
            idx = np.where(y_true_classes == i)[0]
            acc = accuracy_score(y_true_classes[idx], y_pred_classes[idx]) if len(idx) > 0 else 0
            class_name = self.label_map.get(i, f"Class {i}")
            print(f"  {class_name}: {acc:.4f}")