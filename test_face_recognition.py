import face_recognition
import cv2
import numpy as np
import os
import warnings
import sys
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from keras.models import load_model

# 环境配置
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if sys.platform == 'linux':
    os.environ['DISPLAY'] = ':0'

# ---------------------- 加载你训练好的模型 ----------------------
emotion_labels = ["生气", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性"]
MODEL_PATH = "my_emotion_model.h5"  # 你刚生成的模型文件

# 加载模型（必须和训练代码同目录）
try:
    model = load_model(MODEL_PATH)
    print(f"✅ 成功加载自定义训练模型：{MODEL_PATH}")
    print(f"📈 模型验证集准确率约61.48%，可正常识别不同情绪")
except FileNotFoundError:
    print(f"❌ 未找到模型文件：{MODEL_PATH}")
    print(f"⚠️  请确认模型文件和本程序在同一目录！")
    sys.exit(1)
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    sys.exit(1)


# ---------------------- 核心识别函数 ----------------------
def detect_emotion(image_path):
    """人脸情绪识别核心函数（适配你训练的模型）"""
    if not os.path.exists(image_path):
        return "错误：图片文件不存在", None, 0

    try:
        # 读取图片
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)

        # 检测人脸（face_recognition库）
        face_locations = face_recognition.face_locations(img, model="hog")
        if not face_locations:
            return "未检测到人脸", pil_img, 0

        # 处理第一张人脸（和训练时的预处理完全一致）
        top, right, bottom, left = face_locations[0]
        face = img[top:bottom, left:right]

        # 转为48x48灰度图 + 归一化（必须和训练时一致！）
        face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48)) / 255.0
        input_data = np.expand_dims(np.expand_dims(face_resized, axis=0), axis=-1)

        # 预测情绪
        pred = model.predict(input_data, verbose=0)[0]
        emotion_idx = np.argmax(pred)
        emotion = emotion_labels[emotion_idx]
        confidence = round(float(pred[emotion_idx]) * 100, 2)

        # 打印调试信息（看各情绪概率）
        print(f"\n🔍 图片：{os.path.basename(image_path)}")
        print(f"   情绪概率分布（%）：{[round(p * 100, 2) for p in pred]}")
        print(f"   识别结果：{emotion}（置信度{confidence}%）")

        return f"{emotion} (置信度：{confidence}%)", pil_img, len(face_locations)

    except Exception as e:
        error_msg = f"识别失败：{str(e)}"
        print(f"❌ {error_msg}")
        return error_msg, None, 0


# ---------------------- GUI界面 ----------------------
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸情绪识别工具（自定义训练模型）")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # 强制窗口显示
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        # 界面布局
        ttk.Label(
            root,
            text="人脸情绪识别工具（自定义训练模型）",
            font=("微软雅黑", 18, "bold")
        ).pack(pady=10)

        ttk.Button(
            root,
            text="选择图片文件",
            command=self.select_image,
            width=20
        ).pack(pady=5)

        # 图片显示区
        self.img_frame = ttk.Frame(root, width=700, height=400, relief=tk.SUNKEN, borderwidth=2)
        self.img_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.img_frame.pack_propagate(False)
        self.img_label = ttk.Label(self.img_frame, text="请选择图片", anchor=tk.CENTER)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # 结果显示区
        self.result_label = ttk.Label(
            root,
            text="等待选择图片...",
            font=("微软雅黑", 12),
            justify=tk.LEFT
        )
        self.result_label.pack(pady=10, padx=20, anchor=tk.W)

    def select_image(self):
        """选择图片并识别情绪"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("所有文件", "*.*")
            ],
            initialdir=os.path.expanduser("~")
        )

        if not file_path:
            return

        # 执行识别
        emotion_result, pil_img, face_count = detect_emotion(file_path)

        # 更新结果
        result_text = (
            f"图片：{os.path.basename(file_path)}\n"
            f"检测到人脸：{face_count} 个\n"
            f"情绪识别结果：{emotion_result}"
        )
        self.result_label.config(text=result_text)

        # 显示图片
        if pil_img:
            frame_width = self.img_frame.winfo_width()
            frame_height = self.img_frame.winfo_height()
            scale = min(frame_width / pil_img.width, frame_height / pil_img.height, 1.0)
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img_resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img_resized)
            self.img_label.config(image=self.tk_img, text="")
        else:
            self.img_label.config(text="图片加载失败", image="")


# ---------------------- 程序入口 ----------------------
if __name__ == "__main__":
    print("🔍 支持的情绪标签：", emotion_labels)
    print("🚀 启动人脸情绪识别工具...")

    try:
        root = tk.Tk()
        app = EmotionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"❌ GUI启动失败：{str(e)}")
        # 命令行备用模式
        print("\n📌 启用命令行模式")
        while True:
            img_path = input("请输入图片路径（输入q退出）：")
            if img_path.lower() == 'q':
                break
            if os.path.exists(img_path):
                emotion, _, face_count = detect_emotion(img_path)
                print(f"👉 检测到人脸：{face_count} 个 | 识别结果：{emotion}")
            else:
                print("❌ 图片文件不存在，请重新输入！")