# train_baseline_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

def train_baseline(X, y, model_type="mlp", output_dir="outputs/baseline"):
    # Flatten time series data: (N, T, F) -> (N, T*F)
    X_flat = X.reshape(X.shape[0], -1)
    y_int = np.argmax(y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_int, test_size=0.2, random_state=42)

    if model_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)
    elif model_type == "linear":
        model = LogisticRegression(max_iter=300)
    elif model_type == "svm":
        model = SVC()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/{model_type}_model.pkl")

    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    with open(f"{output_dir}/{model_type}_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    print(f"[{model_type.upper()}] Accuracy: {acc:.4f}")
    return model
