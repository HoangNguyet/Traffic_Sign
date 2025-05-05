import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk
import math

# --- Constants ---
MIN_SIGN_AREA = 150
MAX_SIGN_AREA = 50000
# Loại bỏ các ngưỡng hình học chung không cần thiết nữa nếu dùng approxPolyDP
# ASPECT_RATIO_MIN = 0.5
# ASPECT_RATIO_MAX = 1.5
# SOLIDITY_MIN = 0.4
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.75 # Ngưỡng độ tròn để xác định là hình tròn
CONFIDENCE_THRESHOLD = 0.70 # Giảm nhẹ để thử nghiệm
RESIZE_DIM = (32, 32)

# --- Load Model and Labels (Giữ nguyên) ---
try:
    model = load_model("model_26.h5")
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


# --- Image Processing Functions (Giữ nguyên returnHSV, create_binary_mask) ---
def returnHSV(img):
    """Chuyển đổi ảnh sang HSV và làm mờ nhẹ."""
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu (điều chỉnh nếu cần dựa trên thử nghiệm)
low_thresh_red1, high_thresh_red1 = (165, 80, 80), (179, 255, 255) # Dải đỏ 1 (tăng nhẹ Saturation/Value min)
low_thresh_red2, high_thresh_red2 = (0, 80, 80), (10, 255, 255)     # Dải đỏ 2
low_thresh_blue, high_thresh_blue = (95, 80, 80), (130, 255, 255)  # Dải xanh dương
low_thresh_yellow, high_thresh_yellow = (18, 80, 80), (35, 255, 255) # Dải vàng

def create_binary_mask(hsv_img):
    """Tạo ảnh nhị phân kết hợp cho các màu đỏ, xanh, vàng."""
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)

    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel) # Thêm OPEN để loại bỏ nhiễu tốt hơn

    return combined_mask

# --- Preprocessing for Model (Giữ nguyên) ---
def preprocessing(img):
    """Tiền xử lý ảnh đầu vào cho mô hình CNN."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    resized = cv.resize(equalized, RESIZE_DIM)
    normalized = resized / 255.0
    # Nếu dùng Z-score khi huấn luyện, áp dụng ở đây
    return normalized

# --- Prediction Function (Giữ nguyên) ---
def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy cho ảnh biển báo đã tiền xử lý."""
    if sign_image is None or sign_image.size == 0:
        return -1, 0.0

    try:
        processed_img = preprocessing(sign_image)
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0.0

# --- Shape Identification Function (NEW/REVISED) ---
def identify_shape(contour):
    """Xác định hình dạng của contour (tròn, tam giác, chữ nhật, bát giác, hoặc khác)."""
    shape = "unknown"
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return shape # Không thể xấp xỉ nếu chu vi là 0

    # Xấp xỉ contour bằng đa giác
    # Epsilon là khoảng cách tối đa từ contour gốc đến contour xấp xỉ.
    # Giá trị 0.03 * perimeter là một giá trị thường dùng, có thể cần điều chỉnh.
    epsilon = 0.03 * perimeter
    approx = cv.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Tính độ tròn
    area = cv.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

    # Xác định hình dạng dựa trên số đỉnh và độ tròn
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        # Có thể là vuông hoặc chữ nhật (hoặc thoi)
        shape = "rectangle" # Giả định là chữ nhật/vuông cho biển báo
    elif num_vertices == 8:
        shape = "octagon"
    elif num_vertices > 8: # Nếu có nhiều đỉnh và độ tròn cao -> hình tròn
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE:
             shape = "circle"
    # Bạn có thể thêm các trường hợp khác nếu cần (vd: hình thoi)

    # print(f"Vertices: {num_vertices}, Circularity: {circularity:.2f}, Detected Shape: {shape}") # Để debug
    return shape


# --- Sign Detection Function (REVISED) ---
def findSigns(frame):
    """Phát hiện và phân loại biển báo trong ảnh."""
    hsv = returnHSV(frame)
    binary_mask = create_binary_mask(hsv)

    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detected_signs = []
    output_frame = frame.copy()

    for c in contours:
        area = cv.contourArea(c)

        # 1. Lọc theo diện tích
        if MIN_SIGN_AREA < area < MAX_SIGN_AREA:
            # 2. Xác định hình dạng
            shape = identify_shape(c)

            # 3. Chỉ xử lý các hình dạng mong muốn (tam giác, chữ nhật, bát giác, tròn)
            if shape in ["triangle", "rectangle", "octagon", "circle"]:
                x, y, w, h = cv.boundingRect(c)

                # 4. Cắt vùng ảnh nghi ngờ (với lề)
                padding = 5
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                sign = frame[y1:y2, x1:x2]

                if sign.size > 0:
                    # 5. Dự đoán và lấy độ tin cậy
                    label_index, confidence = predict(sign)

                    # 6. Lọc theo ngưỡng tin cậy
                    if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                        label_text = labelToText.get(label_index, f"Unknown ({label_index})")
                        display_text = f"{label_text} ({shape}, {confidence:.2f})" # Thêm tên hình dạng vào text

                        detected_signs.append({'box': (x, y, w, h), 'text': display_text, 'y_pos': y})

    # Sắp xếp và vẽ kết quả (giống như trước)
    detected_signs.sort(key=lambda item: item['y_pos'])
    last_text_y = -100
    text_gap = 20

    for sign_info in detected_signs:
        x, y, w, h = sign_info['box']
        display_text = sign_info['text']
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_x = x
        text_y = y - 10
        if text_y < 10: text_y = y + h + 15
        if abs(text_y - last_text_y) < text_gap: text_y = last_text_y + text_gap

        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Giảm cỡ chữ một chút
        cv.rectangle(output_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
        cv.putText(output_frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        last_text_y = text_y

    return output_frame


# --- GUI Class (Giữ nguyên) ---
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        self.root.geometry("800x700")

        self.label = Label(root, text="Upload an image for traffic sign detection", font=("Arial", 14))
        self.label.pack(pady=15)

        self.upload_button = Button(root, text="Choose Image", command=self.load_image, font=("Arial", 12), width=20, height=2)
        self.upload_button.pack(pady=10)

        self.panel_width = 700
        self.panel_height = 500
        self.panel = Label(root, bg="lightgray", width=self.panel_width, height=self.panel_height)
        self.panel.pack(pady=10, padx=10, expand=True, fill="both")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path: return

        try:
            image = cv.imread(file_path)
            if image is None:
                print(f"Error: Could not read image file {file_path}")
                self.label.config(text=f"Error loading image: {file_path}")
                return

            self.root.update_idletasks()
            detected_image = findSigns(image)

            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)

            panel_w = self.panel.winfo_width()
            panel_h = self.panel.winfo_height()
            if panel_w <= 1 or panel_h <= 1:
                 panel_w = self.panel_width
                 panel_h = self.panel_height

            pil_image.thumbnail((panel_w - 20, panel_h - 20), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(pil_image)
            self.panel.config(image=imgtk, width=panel_w, height=panel_h)
            self.panel.image = imgtk
            self.label.config(text="Detection complete. Choose another image.")

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            self.label.config(text=f"Error processing image: {str(e)}")

# --- Main Execution (Giữ nguyên) ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()