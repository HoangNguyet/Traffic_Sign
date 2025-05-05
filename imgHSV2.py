import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk
import math

# --- Constants ---
# Ngưỡng lọc Contour cơ bản
MIN_SIGN_AREA = 100      # Tăng để loại bỏ nhiễu nhỏ tốt hơn
MAX_SIGN_AREA = 100000
MIN_ASPECT_RATIO_EARLY = 0.2  # Loại bỏ các vật quá hẹp/cao ngay từ đầu
MAX_ASPECT_RATIO_EARLY = 5.0  # Loại bỏ các vật quá rộng/thấp ngay từ đầu

# Ngưỡng nhận dạng hình dạng
APPROX_EPSILON_FACTOR = 0.025 # Giảm nhẹ để cố gắng chính xác hơn
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.7 # Nới lỏng một chút
RECTANGLE_ASPECT_RATIO_MIN = 0.3 # Ngưỡng tỷ lệ W/H cho biển báo chữ nhật thực sự
RECTANGLE_ASPECT_RATIO_MAX = 4.0 # Ngưỡng tỷ lệ W/H cho biển báo chữ nhật thực sự

# Ngưỡng phân loại
CONFIDENCE_THRESHOLD = 0.80 # Tăng để yêu cầu độ chắc chắn cao hơn
RESIZE_DIM = (32, 32)

# --- Load Model and Labels (Giữ nguyên) ---
try:
    # Đảm bảo đường dẫn đến file model là chính xác
    model = load_model("model_26.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'model_24.h5' is in the correct directory.")
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
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu (Rất quan trọng - Cần tinh chỉnh dựa trên thử nghiệm với nhiều ảnh!)
# Thử mở rộng nhẹ dải S và V để bắt màu trong các điều kiện ánh sáng khác nhau hơn
low_thresh_red1, high_thresh_red1 = (165, 60, 60), (179, 255, 255)
low_thresh_red2, high_thresh_red2 = (0, 60, 60), (10, 255, 255)
low_thresh_blue, high_thresh_blue = (95, 60, 60), (130, 255, 255)
low_thresh_yellow, high_thresh_yellow = (18, 60, 80), (35, 255, 255)

# *** LƯU Ý QUAN TRỌNG VỀ MÀU TRẮNG/ĐEN ***
# Mã này HIỆN TẠI KHÔNG tìm kiếm màu trắng/đen.
# Để phát hiện biển báo phụ (như biển thời gian), bạn cần:
# 1. Thêm ngưỡng HSV cho màu trắng (ví dụ: S thấp, V cao).
# 2. Vì màu trắng rất phổ biến, bạn cần các bộ lọc hình dạng/kích thước/tỷ lệ CỰC KỲ chặt chẽ
#    hoặc các kỹ thuật xử lý phức tạp hơn để tránh phát hiện sai hàng loạt.

def create_binary_mask(hsv_img):
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)

    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)

    kernel = np.ones((5, 5), np.uint8)
    # Closing trước để nối các vùng màu gần nhau
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    # Opening sau để loại bỏ các chấm nhiễu nhỏ
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)

    return combined_mask

# --- Preprocessing for Model ---
# !!!!! CỰC KỲ QUAN TRỌNG !!!!!
# Hàm này PHẢI giống hệt 100% với các bước tiền xử lý được áp dụng
# cho dữ liệu KHI HUẤN LUYỆN mô hình 'model_24.h5'.
# Kiểm tra kỹ:
# 1. Thứ tự các bước (Grayscale, Equalize, Resize, Normalize có đúng không?)
# 2. Phương pháp chuẩn hóa (Chia cho 255.0 hay trừ Mean / chia Std Dev?)
# 3. Thư viện Resize (OpenCV hay thư viện khác?) và phương pháp nội suy (interpolation).
# Sai khác dù nhỏ ở bước này sẽ làm GIẢM NGHIÊM TRỌNG độ chính xác khi dự đoán.
def preprocessing(img):
    """Tiền xử lý ảnh đầu vào cho mô hình CNN."""
    # 1. Chuyển sang ảnh xám
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 2. Cân bằng Histogram
    equalized = cv.equalizeHist(gray)
    # 3. Resize ảnh về kích thước chuẩn
    resized = cv.resize(equalized, RESIZE_DIM, interpolation=cv.INTER_AREA) # Chỉ định phương pháp nội suy
    # 4. Chuẩn hóa giá trị pixel
    normalized = resized / 255.0
    # Nếu lúc huấn luyện dùng Z-score (trừ mean, chia std), phải áp dụng chính xác ở đây.
    return normalized

# --- Prediction Function (Giữ nguyên) ---
def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy cho ảnh biển báo."""
    if sign_image is None or sign_image.size == 0:
        return -1, 0.0
    try:
        # Tiền xử lý (Resize và chuẩn hóa đã nằm trong preprocessing)
        processed_img = preprocessing(sign_image)
        # Reshape cho Keras (batch_size, height, width, channels)
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)
        # Dự đoán
        prediction = model.predict(img_array, verbose=0) # Thêm verbose=0 để tránh in log thừa
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0.0

# --- Shape Identification Function (REVISED) ---
def identify_shape(contour):
    """Xác định hình dạng dựa trên approxPolyDP và độ tròn."""
    shape = "unknown"
    perimeter = cv.arcLength(contour, True)
    if perimeter < 10: # Bỏ qua các contour quá nhỏ để xấp xỉ
        return shape

    # Xấp xỉ contour bằng đa giác
    epsilon = APPROX_EPSILON_FACTOR * perimeter
    approx = cv.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Tính độ tròn
    area = cv.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

    # Xác định hình dạng
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        shape = "rectangle" # Sẽ kiểm tra aspect ratio sau
    elif num_vertices == 8:
        # Có thể là bát giác hoặc hình tròn bị xấp xỉ
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE - 0.1: # Nới lỏng hơn cho bát giác/tròn 8 đỉnh
             shape = "octagon" # Ưu tiên bát giác nếu đúng 8 đỉnh và tròn
        else:
             shape = "octagon_like" # Hoặc đánh dấu là giống bát giác
    elif num_vertices >= 7: # Nới lỏng: Chấp nhận 7 đỉnh trở lên cho hình tròn
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE:
             shape = "circle"

    # print(f"Vertices: {num_vertices}, Circularity: {circularity:.2f}, Approx Shape: {shape}") # Debug
    return shape

# --- Sign Detection Function (REVISED) ---
def findSigns(frame):
    """Phát hiện và phân loại biển báo trong ảnh."""
    hsv = returnHSV(frame)
    binary_mask = create_binary_mask(hsv)
    # cv.imshow("Binary Mask", binary_mask) # Bỏ comment để debug mask màu

    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detected_signs = []
    output_frame = frame.copy()

    for c in contours:
        area = cv.contourArea(c)

        # 1. Lọc theo diện tích
        if MIN_SIGN_AREA < area < MAX_SIGN_AREA:
            x, y, w, h = cv.boundingRect(c)

            # 2. Lọc sớm theo tỷ lệ khung hình cơ bản
            aspect_ratio_early = float(w) / h if h > 0 else 0
            if not (MIN_ASPECT_RATIO_EARLY <= aspect_ratio_early <= MAX_ASPECT_RATIO_EARLY):
                continue # Bỏ qua các hình quá dài hoặc quá rộng

            # 3. Xác định hình dạng chi tiết
            shape = identify_shape(c)

            # 4. Lọc dựa trên hình dạng đã xác định
            is_valid_shape = False
            if shape == "triangle":
                is_valid_shape = True # Tam giác thường ổn
            elif shape == "rectangle":
                # Kiểm tra tỷ lệ khung hình chặt hơn cho hình chữ nhật
                aspect_ratio_rect = float(w) / h if h > 0 else 0
                if RECTANGLE_ASPECT_RATIO_MIN <= aspect_ratio_rect <= RECTANGLE_ASPECT_RATIO_MAX:
                    is_valid_shape = True
            elif shape in ["octagon", "circle"]:
                 is_valid_shape = True # Đã kiểm tra độ tròn trong identify_shape

            if is_valid_shape:
                # 5. Cắt vùng ảnh nghi ngờ (với lề nhỏ)
                padding = 5
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                sign_roi = frame[y1:y2, x1:x2]

                if sign_roi.size > 0:
                    # 6. Dự đoán và lấy độ tin cậy
                    label_index, confidence = predict(sign_roi)

                    # 7. Lọc theo ngưỡng tin cậy
                    if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                        label_text = labelToText.get(label_index, f"Unknown ({label_index})")
                        # Thêm hình dạng và độ tin cậy vào text hiển thị để debug
                        display_text = f"{label_text} ({shape[:4]} {confidence:.2f})"

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
        if text_y < 15: text_y = y + h + 15 # Đẩy xuống nếu quá gần mép trên
        # Đơn giản hóa việc chống chồng lấp: nếu quá gần, chỉ cần đẩy xuống một khoảng cố định
        if text_y < last_text_y + text_gap:
             text_y = last_text_y + text_gap

        # Vẽ text với nền
        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(output_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1) # Nền đen
        cv.putText(output_frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Chữ trắng

        last_text_y = text_y # Cập nhật vị trí y cuối cùng

    return output_frame


# --- GUI Class (Giữ nguyên - Đã khá tốt) ---
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        self.root.geometry("850x750") # Tăng kích thước một chút

        self.label = Label(root, text="Upload an image for traffic sign detection", font=("Arial", 14))
        self.label.pack(pady=15)

        self.upload_button = Button(root, text="Choose Image", command=self.load_image, font=("Arial", 12), width=20, height=2)
        self.upload_button.pack(pady=10)

        # Sử dụng Frame để chứa Label ảnh, giúp kiểm soát padding tốt hơn
        self.image_frame = tk.Frame(root, bg="lightgray", bd=1, relief="sunken")
        self.image_frame.pack(pady=10, padx=10, expand=True, fill="both")

        # Label hiển thị ảnh bên trong Frame
        self.panel = Label(self.image_frame, bg="lightgray")
        self.panel.pack(expand=True, fill="both", padx=5, pady=5)

        # Label trạng thái nhỏ ở dưới
        self.status_label = Label(root, text="", font=("Arial", 10), fg="gray")
        self.status_label.pack(pady=5, side="bottom")


    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path: return

        try:
            self.status_label.config(text="Loading image...")
            self.root.update_idletasks()
            image = cv.imread(file_path)
            if image is None:
                self.status_label.config(text=f"Error: Could not read image file.")
                return

            self.status_label.config(text="Processing...")
            self.root.update_idletasks()
            detected_image = findSigns(image)

            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)

            # Lấy kích thước của frame chứa ảnh để tính toán resize
            # Trừ đi padding/border của frame/label nếu có
            frame_w = self.image_frame.winfo_width() - 15
            frame_h = self.image_frame.winfo_height() - 15
            if frame_w <= 1 or frame_h <= 1: # Fallback nếu frame chưa được vẽ
                 frame_w = 700
                 frame_h = 500

            pil_image.thumbnail((frame_w, frame_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(pil_image)
            self.panel.config(image=imgtk)
            self.panel.image = imgtk # Giữ tham chiếu
            self.label.config(text="Detection complete. Choose another image.")
            self.status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")


        except Exception as e:
            print(f"An error occurred during processing: {e}")
            self.label.config(text="An error occurred.")
            self.status_label.config(text=f"Error: {str(e)}")

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()