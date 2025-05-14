import cv2 as cv # Thư viện OpenCV cho xử lý ảnh
import numpy as np # Thư viện NumPy cho xử lý mảng (ảnh)
from keras.models import load_model # type: ignore    # Hàm tải mô hình Keras đã huấn luyện
from tkinter import Tk, filedialog, Button, Label # Thư viện Tkinter cho giao diện đồ họa
from PIL import Image, ImageTk # Thư viện Pillow (PIL) để xử lý và hiển thị ảnh trên Tkinter
import tkinter as tk
import math

# --- Constants ---
#các giá trị này được dùng để điều khiển quá trình xử lý và lọc
# Lọc cơ bản
# Tác động: nếu tăng có thể bỏ lỡ các biển báo nhỏ nhưng nếu giảm có thể gây nhiễu do những vật có cùng màu và nhỏ
MIN_SIGN_AREA = 150      # Ngưỡng diện tích tối thiểu (pixel) của contour được xem xét => loại bỏ nhiễu nhỏ hoặc vùng màu rất nhỏ

#Tác động: nếu giảm thì loại bỏ những biển báo lớn còn khi tăng thì có thể lấy nhầm những vật lớn mà không phải biển báo
MAX_SIGN_AREA = 80000    # Ngưỡng diện tích tối đa => loại bỏ vùng màu lớn bất thường

# Ngưỡng nhận dạng hình dạng
#Tác động: Nếu tăng thì xác định ít đỉnh hơn còn giản thì xác định được nhiều đỉnh hơn, bám sát hơn. => ảnh hưởng tới việc xác định số đỉnh
APPROX_EPSILON_FACTOR = 0.03 # Hệ số cho approxPolyDP (ảnh hưởng đến số đỉnh tìm được)

# Mục đích: Lọc các hình gần tròn.
# Tác động: Tăng -> yêu cầu hình dạng tròn hơn. Giảm -> chấp nhận hình méo hơn.
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.2 # Ngưỡng độ tròn tối thiểu để xác định một contour có thể là hình tròn

# Mục đích: Loại bỏ các hình có nhiều lỗ hổng hoặc lõm sâu bên trong bao lồi (ví dụ: chữ U, chữ C). Biển báo thật thường khá "đặc".
# Tác động: Tăng -> yêu cầu hình dạng đặc hơn. Giảm -> chấp nhận hình có lỗ/lõm nhiều hơn.
SOLIDITY_MIN_THRESHOLD = 0.65 # <<< NGƯỠNG ĐỘ ĐẶC TỐI THIỂU (loại bỏ hình có lỗ/không liền mạch)

# Ngưỡng tỷ lệ khung hình (Aspect Ratio = Width / Height) - Được kiểm tra SAU KHI xác định hình dạng
# Mục đích: Lọc các hình chữ nhật có tỷ lệ phù hợp với biển báo (không quá dài hoặc quá dẹt).
RECTANGLE_ASPECT_RATIO_MIN = 1.0 # Cho phép chữ nhật hơi dọc/ngang
RECTANGLE_ASPECT_RATIO_MAX = 2.0 # Giới hạn trên cho chữ nhật

# Mục đích: Lọc các hình tam giác có tỷ lệ cân đối.
TRIANGLE_ASPECT_RATIO_MIN = 0.6 # Tam giác thường không quá dẹt/cao
TRIANGLE_ASPECT_RATIO_MAX = 1.4

# Mục đích: Đảm bảo các hình tròn/bát giác không quá méo.
CIRCLE_OCTAGON_ASPECT_RATIO_MIN = 0.6 # Tỷ lệ chặt hơn cho hình tròn/bát giác
CIRCLE_OCTAGON_ASPECT_RATIO_MAX = 1.2

# Mục đích: Xác định quy mô cục bộ để tính toán độ sáng trung bình/trọng số.
# Tác động: Lớn -> thích ứng chậm hơn với thay đổi ánh sáng. Nhỏ -> nhạy hơn với chi tiết cục bộ.
# Ngưỡng Adaptive Threshold (Làm chặt hơn để giảm nhiễu từ ảnh xám)
ADAPTIVE_BLOCK_SIZE = 9 # Kích thước vùng lân cận (phải lẻ)

# Tác động: Tăng -> ngưỡng chặt hơn (ít vùng trắng hơn, cần tương phản mạnh hơn). Giảm -> ngưỡng lỏng hơn (nhiều vùng trắng hơn, nhạy hơn với tương phản yếu).
ADAPTIVE_C = 7          # <<< Hằng số trừ đi (càng lớn càng chặt)

# Ngưỡng phân loại (QUAN TRỌNG - Tăng cao để giảm sai sót)
CONFIDENCE_THRESHOLD = 0.5 # <<< Ngưỡng tin cậy tối thiểu của mô hình
RESIZE_DIM = (32, 32)   # Kích thước ảnh đầu vào cho mô hình

# --- Load Model and Labels ---
try:
    # Đảm bảo file model nằm trong cùng thư mục hoặc cung cấp đường dẫn đầy đủ
    model = load_model("model_14.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'model_24.h5' is in the correct directory.")
    exit()

# Dictionary ánh xạ chỉ số lớp sang tên biển báo
# Cần đảm bảo danh sách này đầy đủ và khớp với các lớp của mô hình bạn đã huấn luyện
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
    39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicles > 3.5 tons'
    # Thêm các lớp khác nếu cần...
}

# --- Image Processing Functions ---
def returnHSV(img):
    """Chuyển ảnh sang HSV và làm mờ nhẹ."""
    # Làm mờ để giảm nhiễu trước khi chuyển màu
    blur = cv.GaussianBlur(img, (5, 5), 0)
    # Chuyển sang không gian màu HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu HSV (Rất quan trọng - Cần tinh chỉnh dựa trên thử nghiệm!)
# Dải màu đỏ (cần 2 dải do Hue quay vòng)
# low_thresh_red1, high_thresh_red1 = (165, 40, 40), (179, 255, 255)
# low_thresh_red1, high_thresh_red1 = (136, 87, 11), (179, 255, 255)

# low_thresh_red2, high_thresh_red2 = (0, 40, 40), (10, 255, 255)
# Dải màu xanh dương (mở rộng để bắt nhiều sắc thái hơn)
# low_thresh_blue, high_thresh_blue = (100, 150, 40), (130, 255, 255)
# low_thresh_blue, high_thresh_blue = (94,80,2), (120, 255, 255)

# Dải màu vàng (mở rộng)
# low_thresh_yellow, high_thresh_yellow = (15, 192, 147), (22, 255, 255)
# low_thresh_red1, high_thresh_red1 = (165, 80, 80), (179, 255, 255) # Dải đỏ 1 (tăng nhẹ Saturation/Value min)
# low_thresh_red2, high_thresh_red2 = (0, 80, 80), (10, 255, 255)     # Dải đỏ 2
# low_thresh_blue, high_thresh_blue = (95, 80, 80), (130, 255, 255)  # Dải xanh dương
# low_thresh_yellow, high_thresh_yellow = (18, 80, 80), (35, 255, 255) # Dải vàng

low_thresh_red1, high_thresh_red1 = (160, 40, 40), (180, 255, 255) # Dải đỏ 1 (tăng nhẹ Saturation/Value min)
low_thresh_red2, high_thresh_red2 = (0, 70, 70), (25, 255, 255)     # Dải đỏ 2
low_thresh_blue, high_thresh_blue = (90, 70, 50), (130, 255, 255)  # Dải xanh dương
low_thresh_yellow, high_thresh_yellow = (18, 80, 80), (30, 255, 255) # Dải vàng
def create_binary_mask_hsv(hsv_img):
    """Tạo mask nhị phân kết hợp cho các màu đỏ, xanh, vàng từ ảnh HSV."""
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)
    # Kết hợp các mask màu
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)
    # Áp dụng morphology để dọn dẹp mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel) # Lấp lỗ nhỏ
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)  # Loại bỏ chấm nhiễu
    return combined_mask

def create_binary_mask_gray(gray_img):
    """Tạo mask nhị phân từ ảnh xám bằng Adaptive Thresholding (đã làm chặt hơn)."""
    gray_blur = cv.GaussianBlur(gray_img, (5, 5), 0) # Làm mờ nhẹ
    # Ngưỡng thích nghi Gaussian với hằng số C lớn hơn
    thresh_adapt = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    # Dọn dẹp mask bằng morphology mạnh hơn
    kernel_open = np.ones((5, 5), np.uint8) # Kernel lớn cho OPEN
    kernel_close = np.ones((3, 3), np.uint8)# Kernel nhỏ cho CLOSE
    thresh_adapt = cv.morphologyEx(thresh_adapt, cv.MORPH_OPEN, kernel_open)   # Mở trước để loại bỏ nhiễu
    thresh_adapt = cv.morphologyEx(thresh_adapt, cv.MORPH_CLOSE, kernel_close) # Đóng sau để nối đường mảnh
    return thresh_adapt

# --- Preprocessing for Model ---
# !!!!! CỰC KỲ QUAN TRỌNG !!!!!
# Đảm bảo hàm này khớp 100% với các bước tiền xử lý khi huấn luyện mô hình.
def preprocessing(img):
    """Tiền xử lý ảnh đầu vào cho mô hình CNN."""
    # 1. Chuyển sang ảnh xám
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 2. Cân bằng Histogram để tăng tương phản
    equalized = cv.equalizeHist(gray)
    # 3. Resize ảnh về kích thước chuẩn (chỉ định nội suy để nhất quán)
    resized = cv.resize(equalized, RESIZE_DIM, interpolation=cv.INTER_AREA)
    # 4. Chuẩn hóa giá trị pixel về [0, 1]
    normalized = resized / 255.0
    # Nếu dùng Z-score khi huấn luyện, phải áp dụng chính xác ở đây.
    return normalized

# --- Prediction Function ---
def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy cho ảnh biển báo."""
    if sign_image is None or sign_image.size == 0:
        return -1, 0.0 # Ảnh không hợp lệ
    try:
        processed_img = preprocessing(sign_image)
        # Reshape cho Keras (batch_size=1, height=32, width=32, channels=1)
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)
        # Thực hiện dự đoán (verbose=0 để tắt log thừa)
        prediction = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(prediction) # Lấy chỉ số lớp có xác suất cao nhất
        confidence = np.max(prediction)         # Lấy xác suất cao nhất làm độ tin cậy
        return predicted_label, confidence
    except Exception as e:
        # Ghi lại lỗi nếu có vấn đề trong quá trình dự đoán
        print(f"Error during prediction: {e}")
        return -1, 0.0

# --- Shape Identification Function (REVISED - Thêm tính solidity) ---
def identify_shape(contour):
    """Xác định hình dạng (tam giác, chữ nhật, bát giác, tròn) và tính các thuộc tính."""
    shape = "unknown" #GIá trị mặc định
    perimeter = cv.arcLength(contour, True) #tính chu vi với hàm arcLength
    # Bỏ qua các contour quá nhỏ hoặc có chu vi không hợp lệ
    if perimeter == 0: return shape, 0.0, 0.0

    area = cv.contourArea(contour) #tính diện tích dùng hàm contourArea
    # Lọc diện tích nhỏ hơn nữa ở đây để tránh tính toán thừa
    if area < MIN_SIGN_AREA / 2 : return shape, 0.0, 0.0

    # Tính Solidity (độ đặc)
    hull = cv.convexHull(contour) #tìm bao lồi
    hull_area = cv.contourArea(hull) #tính diện tích bao lồi
    solidity = float(area) / hull_area if hull_area > 0 else 0 #tính độ đặc

    # Tính Circularity (độ tròn)
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

    # Xấp xỉ contour bằng đa giác dùng approxPolyDP
    epsilon = APPROX_EPSILON_FACTOR * perimeter
    approx = cv.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Xác định hình dạng dựa trên số đỉnh
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        shape = "rectangle"
    elif num_vertices == 8:
        # Kiểm tra thêm độ tròn cho trường hợp 8 đỉnh (có thể là bát giác hoặc tròn)
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE - 0.15: # Ngưỡng thấp hơn một chút
             shape = "octagon"
    elif num_vertices >= 7: # Chấp nhận 7 đỉnh trở lên là hình tròn nếu đủ tròn
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE:
             shape = "circle"

    # Trả về hình dạng, độ đặc, độ tròn
    return shape, solidity, circularity

# --- Sign Detection Function (REVISED - Tích hợp các bộ lọc chặt chẽ) ---
def findSigns(frame):
    """Phát hiện và phân loại biển báo, áp dụng bộ lọc chặt chẽ hơn."""
    output_frame = frame.copy() # Làm việc trên bản sao
    detected_signs_info = {} # Dictionary để lưu phát hiện và xử lý trùng lặp

    # 1. Xử lý bằng HSV (cho biển báo màu)
    hsv = returnHSV(frame)
    binary_mask_hsv = create_binary_mask_hsv(hsv)
    contours_hsv, _ = cv.findContours(binary_mask_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 2. Xử lý bằng Grayscale + Adaptive Threshold (cho biển báo đen trắng/tương phản cao)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    binary_mask_gray = create_binary_mask_gray(gray)
    # cv.imshow("Gray Mask", binary_mask_gray) # Bỏ comment dòng này để debug mask xám
    contours_gray, _ = cv.findContours(binary_mask_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 3. Kết hợp contour từ cả hai phương pháp
    all_contours = contours_hsv + contours_gray

    # 4. Lặp qua và lọc các contour
    for c in all_contours:
        area = cv.contourArea(c)
        # Lọc diện tích ban đầu
        if MIN_SIGN_AREA < area:
            # Xác định hình dạng và các thuộc tính
            shape, solidity, circularity = identify_shape(c) # Lấy solidity

            # <<< BỘ LỌC 1: ĐỘ ĐẶC (Solidity) >>>
            if solidity >= SOLIDITY_MIN_THRESHOLD:
                # <<< BỘ LỌC 2: HÌNH DẠNG ĐÃ BIẾT >>>
                if shape != "unknown":
                    x, y, w, h = cv.boundingRect(c)
                    aspect_ratio = float(w) / h if h > 0 else 0

                    # <<< BỘ LỌC 3: TỶ LỆ KHUNG HÌNH (Aspect Ratio) THEO HÌNH DẠNG >>>
                    shape_aspect_ok = False
                    if shape == "triangle" and TRIANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= TRIANGLE_ASPECT_RATIO_MAX:
                        shape_aspect_ok = True
                    elif shape == "rectangle" and RECTANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= RECTANGLE_ASPECT_RATIO_MAX:
                        shape_aspect_ok = True
                    elif shape in ["circle", "octagon"] and CIRCLE_OCTAGON_ASPECT_RATIO_MIN <= aspect_ratio <= CIRCLE_OCTAGON_ASPECT_RATIO_MAX:
                         shape_aspect_ok = True

                    if shape_aspect_ok:
                        # Nếu vượt qua tất cả bộ lọc hình học, tiến hành cắt và phân loại
                        padding = 5 # Thêm lề nhỏ
                        y1, y2 = max(0, y - padding), min(frame.shape[0], y + h + padding)
                        x1, x2 = max(0, x - padding), min(frame.shape[1], x + w + padding)
                        sign_roi = frame[y1:y2, x1:x2] # Cắt vùng quan tâm (Region of Interest)

                        if sign_roi.size > 0: # Đảm bảo ROI không rỗng
                            # Phân loại bằng mô hình
                            label_index, confidence = predict(sign_roi)

                            # <<< BỘ LỌC 4: ĐỘ TIN CẬY (Confidence) >>>
                            if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                                # Xử lý tránh vẽ trùng lặp lên cùng 1 biển báo
                                center_x, center_y = x + w // 2, y + h // 2
                                is_duplicate = False
                                for center_key in list(detected_signs_info.keys()): # Dùng list để có thể xóa khi lặp
                                    dist_sq = (center_x - center_key[0])**2 + (center_y - center_key[1])**2
                                    old_w = detected_signs_info[center_key]['box'][2]
                                    old_h = detected_signs_info[center_key]['box'][3]
                                    # Ngưỡng khoảng cách dựa trên kích thước box lớn hơn
                                    threshold_dist_sq = ((max(w, old_w)/4)**2 + (max(h, old_h)/4)**2)

                                    if dist_sq < threshold_dist_sq:
                                        # Nếu trùng, chỉ giữ lại cái có confidence cao hơn
                                        if confidence > detected_signs_info[center_key]['confidence']:
                                            del detected_signs_info[center_key] # Xóa cái cũ
                                        else:
                                            is_duplicate = True # Bỏ qua cái mới này
                                        break # Đã xử lý trùng lặp, thoát vòng lặp key

                                if not is_duplicate:
                                    # Nếu không trùng hoặc đã thay thế cái cũ, thêm thông tin
                                    label_text = labelToText.get(label_index, f"U:{label_index}") # Dùng U: nếu không có tên
                                    display_text = f"{label_text} ({confidence:.2f})"
                                    detected_signs_info[(center_x, center_y)] = {
                                        'box': (x, y, w, h),
                                        'text': display_text,
                                        'y_pos': y, # Dùng để sắp xếp vẽ
                                        'confidence': confidence
                                    }

    # 5. Vẽ kết quả lên ảnh
    detected_signs = list(detected_signs_info.values())
    detected_signs.sort(key=lambda item: item['y_pos']) # Sắp xếp theo chiều dọc

    last_text_y = -100 # Theo dõi vị trí y của text cuối cùng
    text_gap = 20      # Khoảng cách tối thiểu giữa các dòng text

    for sign_info in detected_signs:
        x, y, w, h = sign_info['box']
        display_text = sign_info['text']

        # Vẽ bounding box màu xanh
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tính toán vị trí vẽ text (phía trên box)
        text_x = x
        text_y = y - 10
        if text_y < 15: text_y = y + h + 15 # Nếu quá gần mép trên, vẽ xuống dưới

        # Đơn giản hóa chống chồng lấp: đẩy xuống nếu quá gần text trước
        if text_y < last_text_y + text_gap:
             text_y = last_text_y + text_gap

        # Lấy kích thước text để vẽ nền
        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Vẽ nền đen cho text
        cv.rectangle(output_frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
        # Vẽ text màu trắng
        cv.putText(output_frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        last_text_y = text_y # Cập nhật vị trí y cuối cùng

    # Nếu bạn muốn dừng lại để xem mask xám khi debug:
    # cv.waitKey(0)
    # cv.destroyWindow("Gray Mask") # Đóng cửa sổ mask

    return output_frame

# --- GUI Class ---
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        # Tăng kích thước cửa sổ một chút
        self.root.geometry("900x800")

        # Label tiêu đề
        self.label = Label(root, text="Upload an image for traffic sign detection", font=("Arial", 14))
        self.label.pack(pady=15)

        # Nút bấm
        self.upload_button = Button(root, text="Choose Image", command=self.load_image, font=("Arial", 12), width=20, height=2)
        self.upload_button.pack(pady=10)

        # Frame chứa ảnh (để có padding và border)
        self.image_frame = tk.Frame(root, bg="lightgray", bd=1, relief="sunken")
        self.image_frame.pack(pady=10, padx=10, expand=True, fill="both")

        # Label hiển thị ảnh bên trong Frame
        self.panel = Label(self.image_frame, bg="lightgray")
        self.panel.pack(expand=True, fill="both", padx=5, pady=5)

        # Label trạng thái ở dưới cùng
        self.status_label = Label(root, text="Please upload an image.", font=("Arial", 10), fg="gray")
        self.status_label.pack(pady=5, side="bottom", fill="x") # fill="x" để căn giữa

    def load_image(self):
        """Hàm xử lý khi người dùng chọn ảnh."""
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.All Files, *.*") ]
        )
        if not file_path:
            self.status_label.config(text="Image selection cancelled.")
            return # Người dùng không chọn file

        try:
            self.status_label.config(text=f"Loading: {file_path.split('/')[-1]}...")
            self.root.update_idletasks() # Cập nhật giao diện ngay

            # Đọc ảnh bằng OpenCV
            image = cv.imread(file_path)
            if image is None:
                self.status_label.config(text=f"Error: Could not read image file.")
                print(f"Error reading file: {file_path}")
                return

            self.status_label.config(text="Processing image...")
            self.root.update_idletasks()

            # <<< Gọi hàm xử lý chính >>>
            detected_image = findSigns(image)

            # Chuyển đổi sang định dạng PIL để hiển thị trên Tkinter
            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)

            # Resize ảnh để vừa với panel hiển thị mà vẫn giữ tỷ lệ
            frame_w = self.image_frame.winfo_width() - 15  # Trừ đi padding/border ước lượng
            frame_h = self.image_frame.winfo_height() - 15
            if frame_w <= 1 or frame_h <= 1: # Fallback nếu frame chưa được vẽ
                 frame_w = 750 # Kích thước dự phòng
                 frame_h = 600

            pil_image.thumbnail((frame_w, frame_h), Image.Resampling.LANCZOS) # Resize giữ tỷ lệ

            # Tạo đối tượng ảnh Tkinter
            imgtk = ImageTk.PhotoImage(pil_image)

            # Cập nhật label hiển thị ảnh
            self.panel.config(image=imgtk)
            self.panel.image = imgtk # <<< Giữ tham chiếu này rất quan trọng!

            # Cập nhật trạng thái
            self.label.config(text="Detection complete. Choose another image.")
            self.status_label.config(text=f"Displayed: {file_path.split('/')[-1]}")

        except Exception as e:
            # Bắt lỗi và hiển thị thông báo
            print(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc() # In chi tiết lỗi ra console
            self.label.config(text="An error occurred during processing.")
            self.status_label.config(text=f"Error: {str(e)}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Tạo cửa sổ chính Tkinter
    root = tk.Tk()
    # Tạo instance của ứng dụng
    app = TrafficSignApp(root)
    # Bắt đầu vòng lặp sự kiện của GUI
    root.mainloop()