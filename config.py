# config.py

# 输入维度（归一化特征长度）
STATE_INPUT_DIM = 17

# 数据预处理参数
REMOVE_STEP = 32
WINDOW_SIZE = 32

# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SAMPLES_PER_CLASS = 5

# 模型参数
EPOCHS = 100
BATCH_SIZE = 64
NUM_CLASSES = 5

# 标签映射
LABEL_MAP = {
    0: "normal",
    1: "engine fault",
    2: "aileron fault",
    3: "rudder fault",
    4: "elevator fault"
}