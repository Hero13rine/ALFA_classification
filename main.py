# main.py

import numpy as np
from model import build_model
from train import train_model
from config import INPUT_DIM, NUM_CLASSES
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# 模拟数据载入（真实可替换）
X = np.random.rand(1000, INPUT_DIM)
y = np.random.randint(0, NUM_CLASSES, size=(1000,))
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# class_weight 计算
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

model = build_model(INPUT_DIM, NUM_CLASSES)
train_model(model, X_train, y_train, X_test, y_test, class_weights=class_weights_dict)
