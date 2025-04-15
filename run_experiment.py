from train import train_with_params

# 定义多个实验配置
experiments = [
    {"batch_size": 32, "epochs": 50, "tag": "bs32_ep50"},
    {"batch_size": 64, "epochs": 100, "tag": "bs64_ep100"},
    {"batch_size": 128, "epochs": 75, "tag": "bs128_ep75"}
]

# 通用配置模板
base_config = {
    "window_size": 32,
    "remove_step": 32,
    "state_input_dim": 17,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "min_samples_per_class": 5,
    "num_classes": 5,
    "label_map": {
        0: "normal",
        1: "engine fault",
        2: "rudder fault",
        3: "elevator fault",
        4: "valid"
    }
}

# 运行每个实验
for exp in experiments:
    config = {**base_config, **exp}
    train_with_params(config)