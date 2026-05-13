import numpy as np
import os
import cv2
import zipfile
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# ===================== 配置 =====================
ZIP_PATH = "archive.zip"
UNZIP_DIR = "fer2013_images"  # 解压后的文件夹
MODEL_SAVE = "my_emotion_model.h5"

# 情绪标签映射（对应文件夹名）
EMOTION_MAP = {
    "angry": 0,  # 生气
    "disgust": 1,  # 厌恶
    "fear": 2,  # 恐惧
    "happy": 3,  # 开心
    "sad": 4,  # 悲伤
    "surprise": 5,  # 惊讶
    "neutral": 6  # 中性
}


# ===================== 解压图片 =====================
def unzip_images():
    if not os.path.exists(UNZIP_DIR):
        print("📦 正在解压 archive.zip 图片数据集...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(UNZIP_DIR)
        print("✅ 解压完成！")


# ===================== 加载图片数据 =====================
def load_image_data():
    unzip_images()

    X = []  # 存储图片数据
    y = []  # 存储情绪标签
    emotion_folders = []

    # 递归查找所有情绪文件夹
    for root, dirs, files in os.walk(UNZIP_DIR):
        for dir_name in dirs:
            if dir_name.lower() in EMOTION_MAP.keys():
                emotion_folders.append(os.path.join(root, dir_name))

    if not emotion_folders:
        print("❌ 未找到情绪分类文件夹！")
        print("⚠️  请确认文件夹名包含：angry/disgust/fear/happy/sad/surprise/neutral")
        exit(1)

    # 加载每个文件夹的图片
    for folder in emotion_folders:
        emotion_name = os.path.basename(folder).lower()
        emotion_label = EMOTION_MAP[emotion_name]
        file_list = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"📸 加载 {emotion_name} 图片：{len(file_list)} 张")

        for img_file in file_list:
            img_path = os.path.join(folder, img_file)
            # 读取为灰度图 + 调整为48x48（和FER2013一致）
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(emotion_label)

    # 数据预处理
    X = np.array(X, dtype=np.float32) / 255.0  # 归一化
    X = np.expand_dims(X, axis=-1)  # 扩展维度 (样本数,48,48,1)
    y = to_categorical(y, num_classes=7)  # 独热编码

    print(f"\n✅ 数据加载完成：")
    print(f"   - 总图片数：{len(X)} 张")
    print(f"   - 数据形状：{X.shape}")
    print(f"   - 标签形状：{y.shape}")

    # 划分训练集/验证集
    return train_test_split(X, y, test_size=0.15, random_state=42)


# ===================== 构建并训练模型 =====================
def build_and_train_model():
    # 加载数据
    X_train, X_val, y_train, y_val = load_image_data()

    # 构建CNN模型
    model = Sequential([
        # 第一层卷积
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # 第二层卷积
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 第三层卷积
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # 全连接层
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(7, activation='softmax')  # 7类情绪
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 打印模型结构
    print("\n🧠 模型结构：")
    model.summary()

    # 训练回调（防止过拟合）
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    # 开始训练
    print("\n🚀 开始训练模型（CPU约30-60分钟）...")
    model.fit(
        X_train, y_train,
        batch_size=32,  # 低配电脑用32，高配可改64
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型
    model.save(MODEL_SAVE)
    print(f"\n✅ 训练完成！模型已保存为：{MODEL_SAVE}")

    # 验证集准确率
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"📈 验证集准确率：{val_acc * 100:.2f}%")


# ===================== 主函数 =====================
if __name__ == "__main__":
    build_and_train_model()