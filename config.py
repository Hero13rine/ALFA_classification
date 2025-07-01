# config.py

# 输入维度（归一化特征长度）
STATE_INPUT_DIM = 17

# 数据预处理参数
REMOVE_STEP = 32
WINDOW_SIZE = 16

# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES_PER_CLASS = 5

# 模型参数
EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 5
# === 让模型学会提前 k_step 预警 ===
LEAD_STEPS      = 8        # 4 Hz × 8 = 2 s 提前量
IGNORE_TAIL     = True     # 末尾不足 k_step 的窗口丢弃
# 标签映射
LABEL_MAP = {
    0: "normal",
    1: "engine fault",
    2: "aileron fault",
    3: "rudder fault",
    4: "elevator fault"
}