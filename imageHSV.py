import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk
import math # Sử dụng cho tính toán aspect ratio

# --- Constants ---
MIN_SIGN_AREA = 100      # Giảm nhẹ để bắt các biển báo nhỏ hơn, nhưn1g tăng nguy cơ nhiễu
MAX_SIGN_AREA = 80000    # Ngưỡng trên để loại bỏ các vùng màu lớn không phải biển báo
ASPECT_RATIO_MIN = 0.5  # Tỷ lệ W/H tối thiểu
ASPECT_RATIO_MAX = 1.5  # Tỷ lệ W/H tối đa (cho phép biển báo hơi chữ nhật)
SOLIDITY_MIN = 0.01     # Ngưỡng tối thiểu cho độ "đặc" của contour
CIRCULARITY_MIN = 0.6  # Ngưỡng tối thiểu cho độ tròn (giảm nhẹ để linh hoạt hơn)
CIRCULARITY_MAX = 1.4  # Ngưỡng tối đa cho độ tròn
CONFIDENCE_THRESHOLD = 0.75 # Ngưỡng tin cậy tối thiểu để hiển thị kết quả (điều chỉnh nếu cần)
RESIZE_DIM = (32, 32)   # Kích thước chuẩn cho mô hình

# --- Load Model and Labels ---
try:
    model = load_model("model_24.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

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

# --- Image Processing Functions ---

def returnHSV(img):
    """Chuyển đổi ảnh sang HSV và làm mờ nhẹ."""
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu (điều chỉnh nếu cần dựa trên thử nghiệm)
low_thresh_red1, high_thresh_red1 = (165, 100, 40), (179, 255, 255) # Dải đỏ 1
low_thresh_red2, high_thresh_red2 = (0, 160, 40), (10, 255, 255)     # Dải đỏ 2 (gần 0 độ)
low_thresh_blue, high_thresh_blue = (100, 150, 40), (130, 255, 255)  # Dải xanh dương
low_thresh_yellow, high_thresh_yellow = (20, 100, 100), (35, 255, 255) # Dải vàng (cho biển báo cảnh báo)

def create_binary_mask(hsv_img):
    """Tạo ảnh nhị phân kết hợp cho các màu đỏ, xanh, vàng."""
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)

    # Kết hợp các mask màu
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)

    # Giảm nhiễu bằng phép toán Morphology Closing
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    # Thêm phép mở để loại bỏ các chấm nhiễu nhỏ (tùy chọn)
    # combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)

    return combined_mask

# --- Preprocessing for Model ---
# QUAN TRỌNG: Hàm này PHẢI giống hệt với hàm tiền xử lý được sử dụng khi HUẤN LUYỆN mô hình.
# Kiểm tra kỹ các bước: chuyển xám, cân bằng histogram, làm mờ, resize, chuẩn hóa (chia 255 hoặc trừ mean/chia std).
def preprocessing(img):
    """Tiền xử lý ảnh đầu vào cho mô hình CNN."""
    # 1. Chuyển sang ảnh xám
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Cân bằng Histogram để cải thiện độ tương phản
    equalized = cv.equalizeHist(gray)

    # 3. Resize ảnh về kích thước đầu vào của mô hình (ví dụ: 32x32)
    resized = cv.resize(equalized, RESIZE_DIM)

    # 4. Chuẩn hóa giá trị pixel về khoảng [0, 1]
    normalized = resized / 255.0

    # (Tùy chọn) Nếu khi huấn luyện bạn đã dùng chuẩn hóa Z-score (trừ mean, chia std):
    # mu = 102.24 # Giá trị mean ví dụ
    # std = 72.12 # Giá trị std ví dụ
    # normalized = (resized - mu) / std

    return normalized

# --- Prediction Function ---
def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy cho ảnh biển báo đã tiền xử lý."""
    if sign_image is None or sign_image.size == 0:
        return -1, 0.0 # Trả về giá trị không hợp lệ nếu ảnh rỗng

    try:
        # Tiền xử lý ảnh (đã được resize trong preprocessing)
        processed_img = preprocessing(sign_image)

        # Reshape lại cho phù hợp với đầu vào của mô hình Keras (batch_size, height, width, channels)
        # Vì ảnh đã là grayscale sau preprocessing, channel là 1
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)

        # Dự đoán bằng mô hình
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)

        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0.0 # Trả về lỗi

# --- Shape Checking Functions ---
def check_shape(contour):
    """Kiểm tra các đặc tính hình học của contour."""
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False, 0.0, 0.0 # Tránh chia cho 0

    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)

    # Solidity: Tỷ lệ diện tích contour / diện tích bao lồi
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Circularity: Mức độ tròn
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

    # Aspect Ratio (từ bounding box)
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # Kiểm tra các ngưỡng
    is_valid_shape = (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX and
                      solidity >= SOLIDITY_MIN and
                      CIRCULARITY_MIN <= circularity <= CIRCULARITY_MAX)

    # Có thể thêm kiểm tra số đỉnh (approxPolyDP) nếu muốn chặt chẽ hơn nữa
    # approx = cv.approxPolyDP(contour, 0.03 * perimeter, True)
    # num_vertices = len(approx)
    # is_valid_shape = is_valid_shape and (num_vertices >= 3 and num_vertices <= 12) # Ví dụ giới hạn số đỉnh

    return is_valid_shape, aspect_ratio, solidity, circularity


# --- Sign Detection Function ---
def findSigns(frame):
    """Phát hiện và phân loại biển báo trong ảnh."""
    hsv = returnHSV(frame)
    binary_mask = create_binary_mask(hsv)

    # Tìm contours trên ảnh nhị phân tổng hợp
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detected_signs = [] # Lưu trữ các biển báo đã phát hiện để tránh vẽ chồng lấp text

    output_frame = frame.copy() # Vẽ lên bản sao để không thay đổi frame gốc

    for c in contours:
        area = cv.contourArea(c)

        # 1. Lọc theo diện tích
        if MIN_SIGN_AREA < area:
            x, y, w, h = cv.boundingRect(c)

            # 2. Lọc theo hình dạng (tỷ lệ, độ đặc, độ tròn)
            is_valid, aspect_r, solidity, circularity = check_shape(c)

            if is_valid:
                # 3. Cắt vùng ảnh nghi ngờ là biển báo (với một chút lề)
                padding = 5 # Thêm lề nhỏ xung quanh
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                sign = frame[y1:y2, x1:x2]

                if sign.size > 0: # Đảm bảo vùng cắt không rỗng
                    # 4. Dự đoán và lấy độ tin cậy
                    label_index, confidence = predict(sign)

                    # 5. Lọc theo ngưỡng tin cậy
                    if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                        label_text = labelToText.get(label_index, f"Unknown ({label_index})")
                        display_text = f"{label_text} ({confidence:.2f})"

                        # Lưu vị trí để kiểm tra chồng lấp (đơn giản)
                        detected_signs.append({'box': (x, y, w, h), 'text': display_text, 'y_pos': y})

    # Sắp xếp các biển báo theo tọa độ y để vẽ text tránh chồng lấp tốt hơn
    detected_signs.sort(key=lambda item: item['y_pos'])

    # Vẽ kết quả lên ảnh (với xử lý chồng lấp đơn giản)
    last_text_y = -100 # Vị trí y của text cuối cùng đã vẽ
    text_gap = 20      # Khoảng cách tối thiểu giữa các dòng text

    for sign_info in detected_signs:
        x, y, w, h = sign_info['box']
        display_text = sign_info['text']

        # Vẽ bounding box
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tính toán vị trí text, cố gắng tránh chồng lấp
        text_x = x
        text_y = y - 10
        if text_y < 10: # Đảm bảo text không bị vẽ ra ngoài mép trên
             text_y = y + h + 15

        # Nếu quá gần text trước đó, đẩy xuống dưới
        if abs(text_y - last_text_y) < text_gap:
            text_y = last_text_y + text_gap

        # Vẽ text với nền đen để dễ đọc hơn
        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(output_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1) # Nền đen
        cv.putText(output_frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Chữ trắng

        last_text_y = text_y # Cập nhật vị trí text cuối cùng

    return output_frame


# --- GUI Class ---
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        # Tăng kích thước cửa sổ mặc định
        self.root.geometry("800x700")

        # Label hướng dẫn
        self.label = Label(root, text="Upload an image for traffic sign detection", font=("Arial", 14))
        self.label.pack(pady=15)

        # Nút Upload Image
        self.upload_button = Button(root, text="Choose Image", command=self.load_image, font=("Arial", 12), width=20, height=2)
        self.upload_button.pack(pady=10)

        # Khung chứa ảnh (Panel) - Đặt kích thước cố định lớn hơn
        self.panel_width = 700
        self.panel_height = 500
        self.panel = Label(root, bg="lightgray", width=self.panel_width, height=self.panel_height)
        # Sử dụng pack thay vì place để bố cục linh hoạt hơn khi thay đổi kích thước cửa sổ
        self.panel.pack(pady=10, padx=10, expand=True, fill="both") # Cho phép panel mở rộng

        # Label hiển thị trạng thái (ví dụ: "Processing...") - không cần thiết nếu xử lý nhanh
        # self.status_label = Label(root, text="", font=("Arial", 10), fg="gray")
        # self.status_label.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return # Người dùng hủy bỏ

        try:
            # Đọc ảnh gốc bằng OpenCV
            image = cv.imread(file_path)
            if image is None:
                print(f"Error: Could not read image file {file_path}")
                self.label.config(text=f"Error loading image: {file_path}")
                return

            # Xử lý ảnh để phát hiện biển báo
            # self.status_label.config(text="Processing...")
            self.root.update_idletasks() # Cập nhật giao diện trước khi xử lý lâu
            detected_image = findSigns(image)
            # self.status_label.config(text="") # Xóa trạng thái

            # Chuyển đổi ảnh kết quả sang định dạng PIL để hiển thị trên Tkinter
            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)

            # Resize ảnh PIL để vừa với Panel mà vẫn giữ tỷ lệ
            # Lấy kích thước của panel đã pack
            panel_w = self.panel.winfo_width()
            panel_h = self.panel.winfo_height()
            # Nếu panel chưa được vẽ, dùng kích thước cố định ban đầu
            if panel_w <= 1 or panel_h <= 1:
                 panel_w = self.panel_width
                 panel_h = self.panel_height

            pil_image.thumbnail((panel_w - 20, panel_h - 20), Image.Resampling.LANCZOS) # Giữ tỷ lệ, -20 để có chút lề

            # Tạo đối tượng ImageTk
            imgtk = ImageTk.PhotoImage(pil_image)

            # Cập nhật Label để hiển thị ảnh
            self.panel.config(image=imgtk, width=panel_w, height=panel_h) # Cập nhật kích thước panel
            self.panel.image = imgtk # Giữ tham chiếu đến ảnh

            self.label.config(text="Detection complete. Choose another image.")

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            self.label.config(text=f"Error processing image: {str(e)}")
            # self.status_label.config(text="")

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()