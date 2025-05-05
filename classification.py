import cv2
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk

# Load mô hình nhận diện biển báo
model = load_model("model_24.h5")

# Định nghĩa nhãn biển báo
labelToText = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing vehicles over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicles > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing vehicles > 3.5 tons'
}

# Tiền xử lý ảnh
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
    img = cv2.equalizeHist(img)  # Cân bằng histogram
    img = img / 255.0  # Chuẩn hóa về [0, 1]
    img = cv2.resize(img, (32, 32))  # Resize về kích thước model yêu cầu
    img = img.reshape(1, 32, 32, 1)  # Reshape thành (1, 32, 32, 1)
    return img

# Dự đoán biển báo
def predict_traffic_sign(image_path):
    img = cv2.imread(image_path)  # Đọc ảnh
    if img is None:
        return "Không thể đọc ảnh!"
    
    processed_img = preprocess_image(img)  # Tiền xử lý
    prediction = model.predict(processed_img)  # Dự đoán
    predicted_class = np.argmax(prediction)  # Lấy class có xác suất cao nhất
    return labelToText.get(predicted_class, "Unknown Sign")

# Giao diện người dùng
class TrafficSignClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Classifier")
        self.root.geometry("500x400")

        # Label hướng dẫn
        self.label = Label(root, text="Chọn ảnh biển báo giao thông để phân loại", font=("Arial", 12))
        self.label.pack(pady=10)

        # Nút chọn ảnh
        self.upload_button = Button(root, text="Chọn ảnh", command=self.load_and_classify)
        self.upload_button.pack(pady=10)

        # Hiển thị ảnh
        self.image_panel = Label(root)
        self.image_panel.pack(pady=10)

        # Hiển thị kết quả
        self.result_label = Label(root, text="", font=("Arial", 14, "bold"), fg="blue")
        self.result_label.pack(pady=10)

    def load_and_classify(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Hiển thị ảnh
            img = Image.open(file_path)
            img.thumbnail((300, 300))  # Resize để hiển thị
            img_tk = ImageTk.PhotoImage(img)
            self.image_panel.config(image=img_tk)
            self.image_panel.image = img_tk

            # Phân loại và hiển thị kết quả
            result = predict_traffic_sign(file_path)
            self.result_label.config(text=f"Kết quả: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignClassifierApp(root)
    root.mainloop()