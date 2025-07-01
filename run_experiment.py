"""

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
"""

from train import train_with_params


base_config = dict(
    tag               = "focal_8step",   # 输出目录后缀
    data_dir          = "data",          # CSV 目录
    state_input_dim   = 17,
    window_size       = 16,
    remove_step       = 32,

    # 训练集划分
    train_ratio       = 0.7,
    val_ratio         = 0.15,
    test_ratio        = 0.15,
    min_samples_per_class = 50,

    # 训练超参
    epochs            = 80,
    batch_size        = 128,
    loss_type         = "focal_loss",    # or 'cross_entropy'

    # 类别信息
    num_classes       = 5,
    label_map =  {
    0: "normal",
    1: "engine fault",
    2: "aileron fault",
    3: "rudder fault",
    4: "elevator fault"
},
    fault_class       = 1,               # 评估发动机故障

    # 早期预警阈值
    lead_steps        = 8,    # 4 Hz × 8 = 2 s 提前量
    alarm_thresh      = 0.8,
    stable_steps      = 3,
)

experiments = [
    {"batch_size": 32, "epochs": 100, "tag": "ce_loss", "loss_type": "cross_entropy"},
    {"batch_size": 32, "epochs": 100, "tag": "weighted_ce", "loss_type": "weighted_cross_entropy"},
    {"batch_size": 32, "epochs": 100, "tag": "focal_loss", "loss_type": "focal_loss"},
]

for exp in experiments:
    config = {**base_config, **exp}
    train_with_params(config)
