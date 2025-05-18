from models.train_baseline_model import train_baseline
from utils.data_loader import compute_mean_std_from_labeled_normals, RNN_set_making

# === 加载 & 预处理数据 ===
import glob
data_files = sorted(glob.glob("data/*.csv"))
normal_mean, normal_std = compute_mean_std_from_labeled_normals(data_files, 32, 17)
X, y, _, _ = RNN_set_making(data_files, 32, normal_mean, normal_std, 17, 32)

# === 执行基线模型对比实验 ===
for model_type in ["mlp", "linear", "svm"]:
    train_baseline(X, y, model_type=model_type, output_dir=f"outputs/baseline_{model_type}")
