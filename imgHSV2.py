import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk
import math
# Thêm các import cần thiết cho GUI Threading
from queue import Queue, Empty
import threading
import logging # Sử dụng logging thay print cho lỗi

# --- Constants ---
# Cần được TINH CHỈNH KỸ LƯỠNG thông qua thử nghiệm!

# Lọc Contour cơ bản
MIN_SIGN_AREA = 200      # <<< TĂNG nhẹ, lọc nhiễu tốt hơn
MAX_SIGN_AREA = 70000    # <<< GIẢM nhẹ, loại bỏ vùng lớn bất thường

# Ngưỡng nhận dạng hình dạng
APPROX_EPSILON_FACTOR = 0.025 # <<< GIẢM nhẹ, bám sát hình dạng hơn

# Ngưỡng Hình học (QUAN TRỌNG - Cần tăng để lọc chặt hơn)
SOLIDITY_MIN_THRESHOLD = 0.80 # <<< TĂNG ĐÁNG KỂ, loại bỏ hình không đặc
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.75 # <<< TĂNG ĐÁNG KỂ cho hình tròn/bát giác

# Ngưỡng tỷ lệ khung hình (Aspect Ratio = Width / Height) - SAU KHI xác định hình dạng
RECTANGLE_ASPECT_RATIO_MIN = 0.7 # Cho phép chữ nhật hơi dọc/ngang
RECTANGLE_ASPECT_RATIO_MAX = 2.5 # Nới rộng hơn một chút cho biển chữ nhật dài
TRIANGLE_ASPECT_RATIO_MIN = 0.6
TRIANGLE_ASPECT_RATIO_MAX = 1.5 # Tam giác có thể hơi không đều cạnh
CIRCLE_OCTAGON_ASPECT_RATIO_MIN = 0.7 # <<< Siết chặt hơn, gần 1.0
CIRCLE_OCTAGON_ASPECT_RATIO_MAX = 1.3 # <<< Siết chặt hơn, gần 1.0

# Ngưỡng phân loại (QUAN TRỌNG - Tăng cao để giảm sai sót)
CONFIDENCE_THRESHOLD = 0.90 # <<< TĂNG CAO, chỉ tin những dự đoán chắc chắn
RESIZE_DIM = (32, 32)   # Kích thước ảnh đầu vào cho mô hình

# Kernel size cho Morphology (có thể thử nghiệm 5x5 hoặc 7x7)
MORPH_KERNEL_SIZE = (5, 5)

# --- Load Model and Labels ---
try:
    # Đảm bảo file model nằm trong cùng thư mục hoặc cung cấp đường dẫn đầy đủ
    model = load_model("model_24.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}", exc_info=True)
    logging.error("Please ensure 'model_24.h5' is in the correct directory.")
    exit()

# Dictionary ánh xạ chỉ số lớp sang tên biển báo (giữ nguyên)
labelToText = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    # ... (giữ nguyên phần còn lại của dictionary) ...
    41: 'End of no passing',
    42: 'End no passing vehicles > 3.5 tons'
}

# --- Image Processing Functions ---
def returnHSV(img):
    """Chuyển ảnh sang HSV và làm mờ nhẹ."""
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NGƯỠNG MÀU HSV CỐ ĐỊNH - CỰC KỲ QUAN TRỌNG - PHẢI TINH CHỈNH BẰNG TRACKBARS
# Các giá trị dưới đây CHỈ LÀ VÍ DỤ và có thể KHÔNG TỐT cho ảnh của bạn.
# Bạn BẮT BUỘC phải dùng công cụ trackbar để tìm giá trị phù hợp.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
low_thresh_red1 = (160, 70, 70)     # Dải đỏ 1 (ví dụ)
high_thresh_red1 = (180, 255, 255)
low_thresh_red2 = (0, 70, 70)       # Dải đỏ 2 (ví dụ)
high_thresh_red2 = (10, 255, 255)
low_thresh_blue = (95, 80, 50)      # Dải xanh dương (ví dụ)
high_thresh_blue = (130, 255, 255)
low_thresh_yellow = (18, 80, 80)    # Dải vàng (ví dụ)
high_thresh_yellow = (35, 255, 255)
# Có thể cần thêm ngưỡng cho màu trắng/xám nếu biển báo có nền trắng rõ
# low_thresh_white = (0, 0, 180)
# high_thresh_white = (180, 30, 255)

def create_binary_mask_hsv(hsv_img):
    """Tạo mask nhị phân kết hợp cho các màu quan tâm từ ảnh HSV."""
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)
    # mask_white = cv.inRange(hsv_img, low_thresh_white, high_thresh_white) # Nếu dùng

    # Kết hợp các mask màu cần thiết
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)
    # combined_mask = cv.bitwise_or(combined_mask, mask_white) # Nếu dùng

    # Áp dụng morphology để dọn dẹp mask NGAY TẠI ĐÂY
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    # CLOSE trước để nối các vùng màu gần nhau, lấp lỗ nhỏ
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=1) # iterations=1 or 2
    # OPEN sau để loại bỏ các chấm nhiễu nhỏ còn sót lại
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel, iterations=1)

    return combined_mask

# --- Loại bỏ hàm create_binary_mask_gray ---

# --- Preprocessing for Model (Giữ nguyên) ---
def preprocessing(img):
    """Tiền xử lý ảnh đầu vào cho mô hình CNN."""
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        logging.warning("Preprocessing received an invalid image.")
        return None # Trả về None nếu ảnh không hợp lệ
    try:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        equalized = cv.equalizeHist(gray)
        resized = cv.resize(equalized, RESIZE_DIM, interpolation=cv.INTER_AREA)
        normalized = resized / 255.0
        return normalized
    except cv.error as e:
        logging.error(f"OpenCV error during preprocessing: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"General error during preprocessing: {e}", exc_info=True)
        return None


# --- Prediction Function (Thêm kiểm tra đầu vào) ---
def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy cho ảnh biển báo."""
    if sign_image is None or sign_image.size == 0:
        logging.warning("Prediction received an invalid sign_image.")
        return -1, 0.0 # Ảnh không hợp lệ

    try:
        processed_img = preprocessing(sign_image)
        if processed_img is None: # Kiểm tra kết quả từ preprocessing
             return -1, 0.0

        # Reshape cho Keras (batch_size=1, height, width, channels=1)
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)

        # Thực hiện dự đoán
        prediction = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_label, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return -1, 0.0

# --- Shape Identification Function (REVISED - Tăng cường kiểm tra) ---
def identify_shape(contour):
    """Xác định hình dạng (tam giác, chữ nhật, bát giác, tròn) và tính các thuộc tính."""
    shape = "unknown"
    perimeter = cv.arcLength(contour, True)
    if perimeter < 10: return shape, 0.0, 0.0 # Bỏ qua chu vi quá nhỏ

    area = cv.contourArea(contour)
    # Lọc diện tích nhỏ hơn nữa ở đây để tránh tính toán thừa
    if area < MIN_SIGN_AREA / 2 : return shape, 0.0, 0.0

    # Tính Solidity (độ đặc) - QUAN TRỌNG
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    # Tránh chia cho 0 và lọc solidity thấp ngay lập tức
    if hull_area <= 0: return shape, 0.0, 0.0
    solidity = float(area) / hull_area
    if solidity < SOLIDITY_MIN_THRESHOLD: # <<< Lọc solidity sớm
        return "low_solidity", solidity, 0.0 # Trả về lý do bị loại

    # Tính Circularity (độ tròn) - QUAN TRỌNG
    # Tránh chia cho 0 cho chu vi
    if perimeter <= 0: return shape, solidity, 0.0
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    # Xấp xỉ contour bằng đa giác
    epsilon = APPROX_EPSILON_FACTOR * perimeter
    approx = cv.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Xác định hình dạng dựa trên số đỉnh và độ tròn
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        # Kiểm tra thêm tính lồi cho hình chữ nhật (loại bỏ hình lõm 4 đỉnh)
        if cv.isContourConvex(approx):
            shape = "rectangle"
        # else: shape = "quad_non_convex" # Debug
    elif num_vertices == 8:
        # Yêu cầu độ tròn cao hơn cho bát giác/tròn 8 đỉnh
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE - 0.1: # Ngưỡng chặt hơn chút
             shape = "octagon"
         # else: shape = "8_vertices_low_circ" # Debug
    elif 5 <= num_vertices <= 7 : # Có thể là tròn bị méo hoặc hình khác
         if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE:
              shape = "circle"
         # else: shape = f"{num_vertices}_vertices_low_circ" # Debug
    elif num_vertices > 8: # Nhiều đỉnh thường là hình tròn
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE:
             shape = "circle"
        # else: shape = f">{num_vertices}_vertices_low_circ" # Debug

    # Trả về hình dạng, độ đặc, độ tròn
    return shape, solidity, circularity

# --- Sign Detection Function (REVISED - Chỉ dùng HSV) ---
def findSigns(frame):
    """Phát hiện và phân loại biển báo chỉ dùng ngưỡng HSV và lọc contour."""
    output_frame = frame.copy()
    detected_signs_info = {} # Dictionary để lưu phát hiện

    # 1. Xử lý bằng HSV
    hsv = returnHSV(frame)
    binary_mask_hsv = create_binary_mask_hsv(hsv)
    # cv.imshow("Combined HSV Mask", binary_mask_hsv) # DEBUG: Xem mask HSV

    # 2. Tìm contours trên mask HSV đã được dọn dẹp
    contours, _ = cv.findContours(binary_mask_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 3. Lặp qua và lọc các contour
    for c in contours:
        area = cv.contourArea(c)
        # Lọc diện tích ban đầu (quan trọng)
        if MIN_SIGN_AREA < area < MAX_SIGN_AREA:
            # Xác định hình dạng và các thuộc tính (Đã tích hợp lọc solidity sớm)
            shape, solidity, circularity = identify_shape(c)

            # <<< BỘ LỌC 1: HÌNH DẠNG HỢP LỆ VÀ ĐỘ ĐẶC CAO >>>
            # identify_shape đã lọc solidity, chỉ cần kiểm tra shape != unknown và không phải lý do loại bỏ
            if shape not in ["unknown", "low_solidity"]:
                x, y, w, h = cv.boundingRect(c)
                if h == 0: continue # Tránh chia cho 0
                aspect_ratio = float(w) / h

                # <<< BỘ LỌC 2: TỶ LỆ KHUNG HÌNH (Aspect Ratio) THEO HÌNH DẠNG >>>
                shape_aspect_ok = False
                if shape == "triangle" and TRIANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= TRIANGLE_ASPECT_RATIO_MAX:
                    shape_aspect_ok = True
                elif shape == "rectangle" and RECTANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= RECTANGLE_ASPECT_RATIO_MAX:
                    shape_aspect_ok = True
                elif shape in ["circle", "octagon"]:
                    # Kiểm tra thêm độ tròn chặt chẽ cho circle/octagon ở đây
                    if CIRCULARITY_THRESHOLD_FOR_CIRCLE <= circularity and \
                       CIRCLE_OCTAGON_ASPECT_RATIO_MIN <= aspect_ratio <= CIRCLE_OCTAGON_ASPECT_RATIO_MAX:
                         shape_aspect_ok = True

                if shape_aspect_ok:
                    # Nếu vượt qua tất cả bộ lọc hình học, tiến hành cắt và phân loại
                    padding = 5 # Thêm lề nhỏ
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)

                    # Đảm bảo vùng cắt hợp lệ
                    if y1 < y2 and x1 < x2:
                        sign_roi = frame[y1:y2, x1:x2]

                        if sign_roi.size > 0:
                            # Phân loại bằng mô hình
                            label_index, confidence = predict(sign_roi)

                            # <<< BỘ LỌC 3: ĐỘ TIN CẬY CAO (Confidence) >>>
                            if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                                # Xử lý tránh vẽ trùng lặp (giữ nguyên logic cũ hoặc dùng NMS)
                                center_x, center_y = x + w // 2, y + h // 2
                                is_duplicate = False
                                keys_to_delete = []
                                for center_key, existing_info in detected_signs_info.items():
                                    dist_sq = (center_x - center_key[0])**2 + (center_y - center_key[1])**2
                                    old_w, old_h = existing_info['box'][2], existing_info['box'][3]
                                    # Ngưỡng khoảng cách dựa trên kích thước trung bình / 3
                                    threshold_dist_sq = (( (w + old_w) / 6)**2 + ( (h + old_h) / 6)**2)

                                    if dist_sq < threshold_dist_sq:
                                        if confidence > existing_info['confidence']:
                                            keys_to_delete.append(center_key)
                                        else:
                                            is_duplicate = True
                                        # Không break, kiểm tra tiếp các box khác
                                # Xóa các box bị thay thế
                                for key in keys_to_delete:
                                    if key in detected_signs_info:
                                        del detected_signs_info[key]

                                if not is_duplicate:
                                    # Lưu thông tin phát hiện hợp lệ
                                    label_text = labelToText.get(label_index, f"U:{label_index}")
                                    display_text = f"{label_text} ({confidence:.2f})"
                                    detected_signs_info[(center_x, center_y)] = {
                                        'box': (x, y, w, h),
                                        'text': display_text,
                                        'y_pos': y, # Dùng để sắp xếp vẽ
                                        'confidence': confidence,
                                        'shape': shape # Lưu lại hình dạng để debug nếu cần
                                    }
                    else:
                         logging.warning(f"Invalid ROI calculated at box: {(x,y,w,h)}")


    # 4. Vẽ kết quả lên ảnh (giữ nguyên logic vẽ)
    detected_signs = sorted(list(detected_signs_info.values()), key=lambda item: item['y_pos'])
    last_text_y = -100
    text_gap = 20

    for sign_info in detected_signs:
        x, y, w, h = sign_info['box']
        display_text = sign_info['text']
        # Có thể thêm shape vào text để debug: display_text += f" S:{sign_info['shape']}"
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_x = x
        text_y = y - 10
        if text_y < 15: text_y = y + h + 15
        # Đơn giản hóa chống chồng lấp
        if text_y < last_text_y + text_gap:
             text_y = last_text_y + text_gap
        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1) # Giảm độ dày text
        # Vẽ nền đen cho text
        cv.rectangle(output_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline + 2), (0, 0, 0), -1)
        # Vẽ text màu trắng
        cv.putText(output_frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # Giảm độ dày text
        last_text_y = text_y # Cập nhật vị trí y cuối cùng

    # Debug: Đóng cửa sổ mask nếu mở
    # try: cv.destroyWindow("Combined HSV Mask")
    # except: pass

    return output_frame


# --- GUI Class (Sử dụng threading như đã triển khai ở phiên bản trước) ---
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection (HSV Only)")
        self.root.geometry("900x800")
        self.result_queue = Queue() # Queue cho kết quả từ thread
        self.current_file_path = None # Lưu đường dẫn file đang xử lý

        # Setup logging for GUI
        self.log_handler = logging.StreamHandler() # Hoặc FileHandler
        self.log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(self.log_formatter)
        logging.getLogger().addHandler(self.log_handler) # Thêm handler vào root logger
        logging.getLogger().setLevel(logging.INFO) # Đặt mức log

        # --- Các thành phần GUI ---
        self.label = Label(root, text="Upload an image for traffic sign detection", font=("Arial", 14))
        self.label.pack(pady=15)

        self.upload_button = Button(root, text="Choose Image", command=self.load_image, font=("Arial", 12), width=20, height=2)
        self.upload_button.pack(pady=10)

        self.image_frame = tk.Frame(root, bg="lightgray", bd=1, relief="sunken")
        self.image_frame.pack(pady=10, padx=10, expand=True, fill="both")

        self.panel = Label(self.image_frame, bg="lightgray")
        self.panel.pack(expand=True, fill="both", padx=5, pady=5)

        self.status_label = Label(root, text="Please upload an image.", font=("Arial", 10), fg="gray")
        self.status_label.pack(pady=5, side="bottom", fill="x")

    # --- Hàm xử lý trong Thread ---
    def process_image_thread(self, file_path):
        try:
            logging.info(f"Processing thread started for: {file_path}")
            image = cv.imread(file_path)
            if image is None:
                logging.error(f"Could not read image file: {file_path}")
                self.result_queue.put(("error", f"Could not read image: {file_path.split('/')[-1]}"))
                return

            # --- Gọi hàm xử lý chính ---
            detected_image = findSigns(image) # Hàm này giờ chỉ dùng HSV

            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)
            logging.info(f"Processing complete for: {file_path}")
            self.result_queue.put(("success", pil_image, file_path))

        except Exception as e:
            logging.error(f"Error in processing thread for {file_path}: {e}", exc_info=True)
            self.result_queue.put(("error", f"Processing error: {str(e)}"))

    # --- Hàm Load Image (Khởi chạy Thread) ---
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            self.status_label.config(text="Image selection cancelled.")
            return

        self.current_file_path = file_path # Lưu lại path
        short_filename = file_path.split('/')[-1]
        self.status_label.config(text=f"Loading: {short_filename}...")
        self.label.config(text="Processing...")
        self.root.update_idletasks()
        self.upload_button.config(state="disabled")
        self.panel.config(image='') # Xóa ảnh cũ

        # Khởi chạy luồng xử lý
        thread = threading.Thread(target=self.process_image_thread, args=(file_path,), daemon=True)
        thread.start()

        # Lên lịch kiểm tra queue kết quả
        self.root.after(100, self.check_result_queue)

    # --- Hàm kiểm tra Queue và cập nhật GUI ---
    def check_result_queue(self):
        try:
            result = self.result_queue.get_nowait()
            status, data = result[0], result[1:]

            # Chỉ cập nhật nếu kết quả là của file hiện tại
            if status == "success" and data[1] == self.current_file_path:
                pil_image, file_path = data
                short_filename = file_path.split('/')[-1]
                logging.info(f"Updating GUI for successful detection: {short_filename}")

                # Resize ảnh để hiển thị
                frame_w = self.image_frame.winfo_width() - 15
                frame_h = self.image_frame.winfo_height() - 15
                if frame_w <= 1 or frame_h <= 1:
                    frame_w, frame_h = 750, 600 # Fallback

                img_w, img_h = pil_image.size
                scale = min(frame_w / img_w, frame_h / img_h, 1.0) # Không phóng to, chỉ thu nhỏ
                new_w, new_h = int(img_w * scale), int(img_h * scale)

                display_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(display_image)
                self.panel.config(image=imgtk)
                self.panel.image = imgtk
                self.label.config(text="Detection complete. Choose another image.")
                self.status_label.config(text=f"Displayed: {short_filename}")
                self.upload_button.config(state="normal")

            elif status == "error": # Hiển thị lỗi bất kể file nào
                error_message = data[0]
                logging.error(f"Received error message: {error_message}")
                self.label.config(text="An error occurred during processing.")
                self.status_label.config(text=f"Error: {error_message}")
                self.upload_button.config(state="normal")

            # Nếu status="success" nhưng không phải file hiện tại, bỏ qua

        except Empty: # Queue rỗng, tiếp tục chờ
            # Kiểm tra xem thread có còn chạy không (cách đơn giản)
            if self.upload_button['state'] == 'disabled':
                 self.root.after(100, self.check_result_queue)
            # Nếu nút đã enabled mà queue rỗng thì thôi
        except Exception as e: # Lỗi khi cập nhật GUI
            logging.error(f"Error processing result from queue or updating GUI: {e}", exc_info=True)
            self.label.config(text="Error displaying result.")
            self.status_label.config(text=f"GUI Update Error: {str(e)}")
            self.upload_button.config(state="normal") # Luôn bật lại nút nếu có lỗi


# --- Main Execution Block ---
if __name__ == "__main__":
    # Cấu hình logging cơ bản ban đầu
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()