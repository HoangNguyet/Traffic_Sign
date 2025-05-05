import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label, Frame
from PIL import Image, ImageTk
import tkinter as tk
import math
import traceback

# --- Constants ---
# Lọc cơ bản
MIN_SIGN_AREA = 150
MAX_SIGN_AREA = 100000
# <<< THÊM BỘ LỌC SỚM >>>
EARLY_ASPECT_RATIO_MIN = 0.15 # Loại bỏ các vật thể quá hẹp/cao ngay từ đầu
EARLY_ASPECT_RATIO_MAX = 6.0  # Loại bỏ các vật thể quá rộng/thấp ngay từ đầu

# Ngưỡng nhận dạng hình dạng
APPROX_EPSILON_FACTOR = 0.03
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.68 # <<< GIẢM ngưỡng tròn
SOLIDITY_MIN_THRESHOLD = 0.50 # <<< GIẢM ngưỡng độ đặc
# Ngưỡng tỷ lệ W/H (Giữ nguyên mức nới lỏng vừa phải)
RECTANGLE_ASPECT_RATIO_MIN = 0.25
RECTANGLE_ASPECT_RATIO_MAX = 4.0
TRIANGLE_ASPECT_RATIO_MIN = 0.5
TRIANGLE_ASPECT_RATIO_MAX = 1.7
CIRCLE_OCTAGON_ASPECT_RATIO_MIN = 0.70
CIRCLE_OCTAGON_ASPECT_RATIO_MAX = 1.35 # Nới lỏng nhẹ

# Ngưỡng Adaptive Threshold (Thử C = 6)
ADAPTIVE_BLOCK_SIZE = 19
ADAPTIVE_C = 6 # <<< THAY ĐỔI C (cần thử nghiệm 4, 5, 6, 7)

# Ngưỡng phân loại (Giữ ở mức thử nghiệm)
CONFIDENCE_THRESHOLD = 0.65 # <<< GIẢM THÊM (Thử nghiệm 0.6 - 0.75)

# Khác
RESIZE_DIM = (32, 32)
DUPLICATE_DISTANCE_FACTOR = 0.3

# --- Load Model and Labels (Giữ nguyên) ---
try:
    model = load_model("model_24.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
    exit()

labelToText = { # Đảm bảo đầy đủ
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
    39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicles > 3.5 tons'
}
print(f"Loaded {len(labelToText)} sign labels.")

# --- Image Processing Functions ---
def returnHSV(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu HSV (Giữ nguyên mức đã mở rộng - có thể cần tinh chỉnh thêm)
low_thresh_red1, high_thresh_red1 = (165, 40, 40), (179, 255, 255)
low_thresh_red2, high_thresh_red2 = (0, 40, 40), (10, 255, 255)
low_thresh_blue, high_thresh_blue = (88, 40, 35), (138, 255, 255)
low_thresh_yellow, high_thresh_yellow = (15, 40, 60), (40, 255, 255)
print("HSV thresholds set (Red1, Red2, Blue, Yellow). Adjust if needed.")

def create_binary_mask_hsv(hsv_img):
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)
    # <<< GIẢM Morphology để tránh nối contour >>>
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=1) # Chỉ 1 lần
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel, iterations=1)  # Chỉ 1 lần
    return combined_mask

def create_binary_mask_gray(gray_img):
    gray_blur = cv.GaussianBlur(gray_img, (5, 5), 0)
    thresh_adapt = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    # <<< GIẢM Morphology >>>
    kernel = np.ones((3, 3), np.uint8)
    thresh_adapt = cv.morphologyEx(thresh_adapt, cv.MORPH_OPEN, kernel, iterations=1)
    thresh_adapt = cv.morphologyEx(thresh_adapt, cv.MORPH_CLOSE, kernel, iterations=1)
    return thresh_adapt

# --- Preprocessing & Prediction (Giữ nguyên) ---
def preprocessing(img):
    # ... (Giữ nguyên) ...
    try:
        if len(img.shape) == 2: gray = img
        elif img.shape[2] == 1: gray = img.reshape(img.shape[0], img.shape[1])
        else: gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        equalized = cv.equalizeHist(gray)
        resized = cv.resize(equalized, RESIZE_DIM, interpolation=cv.INTER_AREA)
        normalized = resized / 255.0
        return normalized
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def predict(sign_image):
    # ... (Giữ nguyên) ...
    if sign_image is None or sign_image.size == 0: return -1, 0.0
    try:
        processed_img = preprocessing(sign_image)
        if processed_img is None: return -1, 0.0
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)
        prediction = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0.0

# --- Shape Identification Function (Giữ nguyên logic, ngưỡng tròn thay đổi ở Constants) ---
def identify_shape(contour):
    # ... (Giữ nguyên) ...
    shape = "unknown"
    perimeter = cv.arcLength(contour, True)
    if perimeter < 30: return shape, 0.0, 0.0
    area = cv.contourArea(contour)
    if area < MIN_SIGN_AREA * 0.8 : return shape, 0.0, 0.0
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
    epsilon = APPROX_EPSILON_FACTOR * perimeter
    approx = cv.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)
    if num_vertices == 3: shape = "triangle"
    elif num_vertices == 4: shape = "rectangle"
    elif num_vertices == 8: shape = "octagon"
    elif num_vertices >= 7 and num_vertices <= 16:
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE: shape = "circle"
    return shape, solidity, circularity

# --- Sign Detection Function (REVISED - Thêm lọc sớm) ---
def findSigns(frame):
    output_frame = frame.copy()
    detected_signs_info = {}

    # === BƯỚC DEBUG: HIỂN THỊ MASK (Bỏ comment nếu cần) ===
    # hsv_debug = returnHSV(frame)
    # mask_hsv_debug = create_binary_mask_hsv(hsv_debug)
    # gray_debug = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # mask_gray_debug = create_binary_mask_gray(gray_debug)
    # cv.imshow("DEBUG HSV Mask", mask_hsv_debug)
    # cv.imshow("DEBUG Gray Mask", mask_gray_debug)
    # cv.waitKey(1) # Cho phép cửa sổ hiển thị ngắn
    # === KẾT THÚC DEBUG MASK ===

    # 1. Xử lý HSV
    hsv = returnHSV(frame)
    binary_mask_hsv = create_binary_mask_hsv(hsv)
    contours_hsv, _ = cv.findContours(binary_mask_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 2. Xử lý Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    binary_mask_gray = create_binary_mask_gray(gray)
    contours_gray, _ = cv.findContours(binary_mask_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 3. Kết hợp contours
    all_contours = contours_hsv + contours_gray

    # === BƯỚC DEBUG: VẼ TẤT CẢ CONTOUR BAN ĐẦU (Bỏ comment nếu cần) ===
    # debug_contours_frame = frame.copy()
    # cv.drawContours(debug_contours_frame, contours_hsv, -1, (0, 0, 255), 1) # HSV màu đỏ
    # cv.drawContours(debug_contours_frame, contours_gray, -1, (255, 0, 0), 1) # Gray màu xanh
    # cv.imshow("DEBUG All Contours", debug_contours_frame)
    # cv.waitKey(1)
    # === KẾT THÚC DEBUG CONTOUR ===

    # 4. Lọc và xử lý từng contour
    for c in all_contours:
        # Lọc diện tích ban đầu
        area = cv.contourArea(c)
        if MIN_SIGN_AREA < area < MAX_SIGN_AREA:
            x, y, w, h = cv.boundingRect(c)
            # Tránh box quá nhỏ
            if w < 10 or h < 10: continue

            # <<< BỘ LỌC SỚM: TỶ LỆ KHUNG HÌNH TỔNG QUÁT >>>
            aspect_ratio_early = float(w) / h
            if not (EARLY_ASPECT_RATIO_MIN <= aspect_ratio_early <= EARLY_ASPECT_RATIO_MAX):
                continue # Bỏ qua nếu quá dài hoặc quá rộng

            # Nếu qua lọc sớm, mới kiểm tra hình dạng chi tiết
            shape, solidity, circularity = identify_shape(c)

            # Lọc độ đặc
            if solidity >= SOLIDITY_MIN_THRESHOLD:
                # Lọc hình dạng đã biết
                if shape != "unknown":
                    # Lọc tỷ lệ khung hình theo hình dạng
                    shape_aspect_ok = False
                    aspect_ratio = float(w) / h # Tính lại cho rõ ràng
                    if shape == "triangle" and TRIANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= TRIANGLE_ASPECT_RATIO_MAX:
                        shape_aspect_ok = True
                    elif shape == "rectangle" and RECTANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= RECTANGLE_ASPECT_RATIO_MAX:
                        shape_aspect_ok = True
                    elif shape in ["circle", "octagon"] and CIRCLE_OCTAGON_ASPECT_RATIO_MIN <= aspect_ratio <= CIRCLE_OCTAGON_ASPECT_RATIO_MAX:
                         shape_aspect_ok = True

                    if shape_aspect_ok:
                        # Cắt ROI
                        padding = 5
                        y1, y2 = max(0, y - padding), min(frame.shape[0], y + h + padding)
                        x1, x2 = max(0, x - padding), min(frame.shape[1], x + w + padding)
                        sign_roi = frame[y1:y2, x1:x2]

                        if sign_roi.size > 0:
                            # Phân loại
                            label_index, confidence = predict(sign_roi)

                            # Lọc độ tin cậy
                            if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                                # Xử lý trùng lặp (Giữ nguyên logic)
                                center_x, center_y = x + w // 2, y + h // 2
                                is_duplicate = False
                                dist_factor_sq = DUPLICATE_DISTANCE_FACTOR**2
                                for center_key in list(detected_signs_info.keys()):
                                    dist_sq = (center_x - center_key[0])**2 + (center_y - center_key[1])**2
                                    old_w = detected_signs_info[center_key]['box'][2]
                                    old_h = detected_signs_info[center_key]['box'][3]
                                    threshold_dist_sq = ((max(w, old_w)**2)*dist_factor_sq + (max(h, old_h)**2)*dist_factor_sq)
                                    if dist_sq < threshold_dist_sq:
                                        if confidence > detected_signs_info[center_key]['confidence']:
                                            del detected_signs_info[center_key]
                                        else:
                                            is_duplicate = True
                                        break
                                if not is_duplicate:
                                    label_text = labelToText.get(label_index, f"L:{label_index}")
                                    display_text = f"{label_text} ({confidence:.2f})"
                                    detected_signs_info[(center_x, center_y)] = {
                                        'box': (x, y, w, h), 'text': display_text, 'y_pos': y, 'confidence': confidence
                                    }

    # 5. Vẽ kết quả (Giữ nguyên)
    detected_signs = list(detected_signs_info.values())
    detected_signs.sort(key=lambda item: item['y_pos'])
    last_text_y = -100
    text_gap = 20
    for sign_info in detected_signs:
        x, y, w, h = sign_info['box']
        display_text = sign_info['text']
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_x = x
        text_y = y - 10
        if text_y < 15: text_y = y + h + 15
        if text_y < last_text_y + text_gap: text_y = last_text_y + text_gap
        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(output_frame, (text_x, text_y - text_height - baseline - 2), (text_x + text_width + 2, text_y + baseline), (0, 0, 0), -1)
        cv.putText(output_frame, display_text, (text_x + 1, text_y - 1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        last_text_y = text_y

    # Đóng cửa sổ debug nếu đã mở
    # cv.destroyAllWindows()

    return output_frame

# --- GUI Class & Main Execution (Giữ nguyên) ---
class TrafficSignApp:
    # ... (Giữ nguyên code GUI) ...
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection - Graduation Project")
        self.root.geometry("900x800")
        top_frame = Frame(root)
        top_frame.pack(pady=10)
        self.label = Label(top_frame, text="Upload an image for traffic sign detection", font=("Arial", 14))
        self.label.pack()
        self.upload_button = Button(top_frame, text="Choose Image", command=self.load_image, font=("Arial", 12, "bold"), width=20, height=2, bg="#DDDDFF")
        self.upload_button.pack(pady=10)
        self.image_frame = Frame(root, bg="gray", bd=2, relief="sunken")
        self.image_frame.pack(pady=10, padx=10, expand=True, fill="both")
        self.panel = Label(self.image_frame, bg="darkgray")
        self.panel.pack(expand=True, fill="both", padx=5, pady=5)
        self.status_label = Label(root, text="Welcome! Please upload an image.", font=("Arial", 10), fg="#333333", bd=1, relief="sunken", anchor='w')
        self.status_label.pack(pady=5, padx=10, side="bottom", fill="x")

    def load_image(self):
        # ... (Giữ nguyên logic load ảnh và gọi findSigns) ...
        try:
            file_path = filedialog.askopenfilename(
                title="Select an Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
            )
            if not file_path:
                self.status_label.config(text="Image selection cancelled.")
                return

            self.status_label.config(text=f"Loading: {os.path.basename(file_path)}...")
            self.root.update_idletasks()
            image = cv.imread(file_path)
            if image is None:
                self.status_label.config(text="Error: Could not read image file.")
                return

            self.status_label.config(text="Processing image... Please wait.")
            self.root.update_idletasks()
            detected_image = findSigns(image)
            self.status_label.config(text="Preparing display...")
            self.root.update_idletasks()

            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)
            frame_w = self.image_frame.winfo_width() - 15
            frame_h = self.image_frame.winfo_height() - 15
            if frame_w <= 1 or frame_h <= 1: frame_w, frame_h = 800, 600
            pil_image.thumbnail((frame_w, frame_h), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil_image)
            self.panel.config(image=imgtk)
            self.panel.image = imgtk
            self.label.config(text="Detection complete. Choose another image.")
            self.status_label.config(text=f"Displayed: {os.path.basename(file_path)}")
        except Exception as e:
            print("--- AN ERROR OCCURRED ---")
            traceback.print_exc()
            self.label.config(text="An error occurred during processing!")
            self.status_label.config(text=f"Error: {type(e).__name__}")


if __name__ == "__main__":
    print("Initializing GUI...")
    root = tk.Tk()
    app = TrafficSignApp(root)
    print("Starting main loop...")
    root.mainloop()
    print("GUI closed.")