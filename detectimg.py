import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog
from PIL import Image
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

# Chuyển đổi sang ảnh HSV
def returnHSV(img):
    # cv.imshow("Original Image", img)
    # cv.waitKey(0)
    
    blur = cv.GaussianBlur(img, (5,5), 0)
    # cv.imshow("Blurred Image", blur)
    # cv.waitKey(0)
    
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    # cv.imshow("HSV Image", hsv)
    # cv.waitKey(0)
    
    return hsv

# Ngưỡng để lọc màu
low_thresh1, high_thresh1 = (165, 100, 40), (179, 255, 255)
low_thresh2, high_thresh2 = (0, 160, 40), (10, 255, 255)
low_thresh3, high_thresh3 = (100, 150, 40), (130, 255, 255)
low_thresh4, high_thresh4 = (25, 100, 100), (35, 255, 255)  # Ngưỡng cho màu vàng

# Tạo ảnh nhị phân dựa trên màu sắc
def binaryImg(img):
    hsv = returnHSV(img)
    b_img_red = cv.inRange(hsv, low_thresh1, high_thresh1) | cv.inRange(hsv, low_thresh2, high_thresh2)
    b_img_blue = cv.inRange(hsv, low_thresh3, high_thresh3)
    # b_img_yellow = cv.inRange(hsv, low_thresh4, high_thresh4)
    # cv.imshow("Red Binary Image", b_img_red)
    # cv.waitKey(0)
    # cv.imshow("Blue Binary Image", b_img_blue)
    # cv.waitKey(0)
    # Giảm nhiễu bằng phép toán đóng (Closing)
    kernel = np.ones((5,5), np.uint8)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_CLOSE, kernel)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_CLOSE, kernel)
    return b_img_red, b_img_blue

# Xử lý ảnh trước khi đưa vào mô hình
def grayscale(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv.equalizeHist(img)
    return img

def preprocessing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("Grayscale Image", gray)
    # cv.waitKey(0)
    
    equalized = cv.equalizeHist(gray)
    # cv.imshow("Equalized Image", equalized)
    # cv.waitKey(0)
    
    img = equalized / 255
    return img

# def preprocessingImage(image, imageSize=32, mu=102.24, std=72.12):
#     image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#     image = cv.resize(image, (imageSize, imageSize))
#     image = (image - mu) / std
#     image = image.reshape(1, imageSize, imageSize, 1)
#     return image

# Dự đoán nhãn biển báo
def predict(sign):
    img = preprocessing(sign)
    img = cv.resize(img, (32, 32))
    # cv.imshow("Preprocessed Image (32x32)", img)
    # cv.waitKey(0)
    
    img = img.reshape(1, 32, 32, 1)
    return np.argmax(model.predict(img))


def is_circular(contour):
    """
    Kiểm tra xem contour có dạng hình tròn hay không.
    """
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 < circularity < 1.2  # Giá trị gần 1 là hình tròn

def is_triangle_or_rectangle(contour):
    """
    Kiểm tra xem contour có phải hình tam giác hoặc hình chữ nhật không.
    """
    approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    if len(approx) == 3:  # Hình tam giác
        return "triangle"
    elif len(approx) == 4:  # Hình chữ nhật/vuông
        return "rectangle"
    return None  # Không phải hình tam giác hay hình chữ nhật

def is_specific_shape(contour):
    """
    Kiểm tra xem contour có phải là hình thoi hoặc hình bát giác không.
    """
    approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    num_vertices = len(approx)

    if num_vertices == 4:
        return "diamond"  # Hình thoi
    elif num_vertices == 8:
        return "octagon"  # Hình bát giác
    return None  # Không phải hình cần tìm

def findSigns(frame):
    b_img_red, b_img_blue = binaryImg(frame)

    contours_red, _ = cv.findContours(b_img_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv.findContours(b_img_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    used_positions = []  # Danh sách lưu vị trí các label đã đặt

    for c in contours_red + contours_blue:
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)

        # Kiểm tra hình dạng
        shape = is_triangle_or_rectangle(c)
        is_circle = is_circular(c)
        specific_shape = is_specific_shape(c)

        # Xác định nếu contour là hình hợp lệ
        if area > 1500 and (is_circle or shape in ["triangle", "rectangle"] or specific_shape in ["diamond", "octagon"]):
            sign = frame[y:y+h, x:x+w]
            label = predict(sign)
            label_text = labelToText.get(label, "Unknown Sign")

            # Xử lý vị trí tránh đè label
            text_x, text_y = x, y - 10  # Vị trí mặc định
            min_distance = 20  # Khoảng cách tối thiểu giữa các label

            # Kiểm tra nếu vị trí này bị trùng
            for (px, py) in used_positions:
                if abs(text_y - py) < min_distance:  # Nếu khoảng cách quá gần
                    text_y += min_distance  # Dời xuống để tránh đè

            used_positions.append((text_x, text_y))  # Lưu vị trí đã dùng

            # Vẽ hình chữ nhật xung quanh biển báo
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, label_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

# Giao diện tải ảnh
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        self.root.geometry("600x600")  # Mở rộng khung để hiển thị tốt hơn

        # Label hướng dẫn
        self.label = Label(root, text="Upload an image", font=("Arial", 12))
        self.label.pack(pady=10)

          # Nút Upload Image
        self.upload_button = Button(root, text="Choose Image", command=self.load_image)
        self.upload_button.pack(pady=10)

        # Khung chứa ảnh (Giữ khoảng trống để in tên biển báo)
        self.panel = Label(root, bg="gray")
        self.panel.pack(pady=10)

        # Label hiển thị tên biển báo
        self.result_label = Label(root, text="", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv.imread(file_path)
            detected_image = findSigns(image)
            detected_image = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)

            # Giữ nguyên kích thước ảnh gốc
            max_width, max_height = 800, 600
            img = Image.fromarray(detected_image)
            img.thumbnail((max_width, max_height))
            img = ImageTk.PhotoImage(img)

            # Cập nhật Label để hiển thị ảnh
            self.panel.config(image=img)
            self.panel.image = img

            # Cập nhật kích thước và căn giữa lại
            self.panel.place(relx=0.5, rely=0.5, anchor="center")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop() 