import cv2 as cv
import numpy as np
# Tắt bớt log TensorFlow/Keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras # Nên import keras từ tensorflow
from keras.models import load_model # type: ignore
import math
import traceback # Để in lỗi chi tiết

# --- Constants (Lấy từ code ảnh, CẦN TINH CHỈNH KỸ LƯỠNG CHO VIDEO) ---
# Lọc cơ bản
MIN_SIGN_AREA = 150      # Ngưỡng diện tích tối thiểu (pixel) - video có thể cần giá trị nhỏ hơn ảnh tĩnh
MAX_SIGN_AREA = 80000    # Ngưỡng diện tích tối đa
# Bộ lọc sớm - loại bỏ hình dạng quá bất thường ngay từ đầu
EARLY_ASPECT_RATIO_MIN = 0.15 # Tỷ lệ W/H tối thiểu
EARLY_ASPECT_RATIO_MAX = 6.0  # Tỷ lệ W/H tối đa

# Ngưỡng nhận dạng hình dạng
APPROX_EPSILON_FACTOR = 0.03 # Hệ số cho approxPolyDP
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.68 # Ngưỡng tròn tối thiểu
SOLIDITY_MIN_THRESHOLD = 0.50 # Ngưỡng độ đặc tối thiểu (có thể cần nới lỏng hơn cho video)
# Ngưỡng tỷ lệ W/H theo hình dạng
RECTANGLE_ASPECT_RATIO_MIN = 0.25
RECTANGLE_ASPECT_RATIO_MAX = 4.0
TRIANGLE_ASPECT_RATIO_MIN = 0.5
TRIANGLE_ASPECT_RATIO_MAX = 1.7
CIRCLE_OCTAGON_ASPECT_RATIO_MIN = 0.70
CIRCLE_OCTAGON_ASPECT_RATIO_MAX = 1.35

# Ngưỡng Adaptive Threshold (Tinh chỉnh Cẩn Thận!)
ADAPTIVE_BLOCK_SIZE = 19 # Kích thước vùng lân cận (phải lẻ)
ADAPTIVE_C = 6          # Hằng số trừ đi (Thử 4-7)

# Ngưỡng phân loại (CỰC KỲ QUAN TRỌNG CHO VIDEO)
CONFIDENCE_THRESHOLD = 0.75 # <<< Tinh chỉnh giá trị này (0.65 - 0.9)

# Khác
RESIZE_DIM = (32, 32)   # Kích thước ảnh đầu vào mô hình
DUPLICATE_DISTANCE_FACTOR = 0.3 # Hệ số khoảng cách để xử lý trùng lặp

# --- Load Model ---
# !!! SỬ DỤNG MODEL CỦA VIDEO SCRIPT !!!
MODEL_PATH = "model_24.h5"
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
    # Không cần compile lại khi chỉ dự đoán
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
    print(f"Ensure '{MODEL_PATH}' exists and is a valid Keras model file.")
    exit()

# --- Label Mapping (Đảm bảo khớp với model_26.h5) ---
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
}
print(f"Loaded {len(labelToText)} sign labels.")

# --- HSV Thresholds (Lấy từ code ảnh, tinh chỉnh nếu cần) ---
# Đỏ
low_thresh_red1, high_thresh_red1 = (165, 40, 40), (179, 255, 255)
low_thresh_red2, high_thresh_red2 = (0, 40, 40), (10, 255, 255)
# Xanh dương
low_thresh_blue, high_thresh_blue = (88, 40, 35), (138, 255, 255) # Dùng ngưỡng rộng hơn
# Vàng
low_thresh_yellow, high_thresh_yellow = (15, 40, 60), (40, 255, 255) # Dùng ngưỡng rộng hơn
print("HSV thresholds set. Adjust based on video conditions.")

# --- Image Processing Helper Functions (Lấy từ code ảnh) ---
def returnHSV(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

def create_binary_mask_hsv(hsv_img):
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)
    # Dùng kernel nhỏ và ít iteration hơn cho video để tránh mất chi tiết nhanh
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel, iterations=1)
    return combined_mask

def create_binary_mask_gray(gray_img):
    gray_blur = cv.GaussianBlur(gray_img, (5, 5), 0)
    thresh_adapt = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    kernel = np.ones((3, 3), np.uint8)
    thresh_adapt = cv.morphologyEx(thresh_adapt, cv.MORPH_OPEN, kernel, iterations=1)
    thresh_adapt = cv.morphologyEx(thresh_adapt, cv.MORPH_CLOSE, kernel, iterations=1)
    return thresh_adapt

def identify_shape(contour):
    # (Giữ nguyên logic từ code ảnh)
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

# --- Preprocessing function (SỬ DỤNG PHIÊN BẢN CỦA VIDEO GỐC - Mean/Std Norm) ---
# !! QUAN TRỌNG: Giả định model_26.h5 được huấn luyện với cách này !!
def preprocessingImageToClassifier(image=None, imageSize=32, mu=102.23982103497072, std=72.11947698025735):
    """Tiền xử lý ảnh cho mô hình (Mean/Std Normalization)."""
    if image is None or image.size == 0:
        # print("Warning: preprocessing received empty image.") # Giảm log
        return None
    try:
        # Chuyển sang Grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
             gray = image
        else: return None

        # Không dùng equalizeHist trừ khi model được huấn luyện cùng nó
        # gray = cv.equalizeHist(gray)

        resized = cv.resize(gray, (imageSize, imageSize), interpolation=cv.INTER_AREA)
        normalized = (resized - mu) / std # Z-score normalization
        reshaped = normalized.reshape(1, imageSize, imageSize, 1)
        return reshaped
    except Exception as e:
        print(f"Error in preprocessingImageToClassifier: {e}")
        # traceback.print_exc()
        return None

# --- Prediction Function (Dùng đúng preprocessing) ---
def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy."""
    if sign_image is None or sign_image.size == 0: return -1, 0.0
    try:
        # Gọi hàm tiền xử lý ĐÚNG của video script
        img_array = preprocessingImageToClassifier(sign_image, imageSize=RESIZE_DIM[0])
        if img_array is None: return -1, 0.0

        prediction = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        # traceback.print_exc()
        return -1, 0.0

# --- NEW Sign Detection Function (Hàm chính để phát hiện) ---
def findSigns(frame):
    """Phát hiện, lọc và phân loại biển báo trong frame."""
    detected_rois = []
    detected_labels = []
    detected_signs_info = {} # Dùng dict để xử lý trùng lặp
    frame_height, frame_width, _ = frame.shape

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

    # 4. Lọc và xử lý từng contour
    for c in all_contours:
        area = cv.contourArea(c)
        if MIN_SIGN_AREA < area < MAX_SIGN_AREA: # Thêm lại MAX_AREA
            x, y, w, h = cv.boundingRect(c)
            if w < 10 or h < 10: continue

            # Lọc sớm Aspect Ratio
            aspect_ratio_early = float(w) / h
            if not (EARLY_ASPECT_RATIO_MIN <= aspect_ratio_early <= EARLY_ASPECT_RATIO_MAX):
                continue

            shape, solidity, circularity = identify_shape(c)

            # Lọc Solidity
            if solidity >= SOLIDITY_MIN_THRESHOLD:
                # Lọc Hình dạng
                if shape != "unknown":
                    # Lọc Aspect Ratio theo hình dạng
                    aspect_ratio = float(w) / h
                    shape_aspect_ok = False
                    if shape == "triangle" and TRIANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= TRIANGLE_ASPECT_RATIO_MAX: shape_aspect_ok = True
                    elif shape == "rectangle" and RECTANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= RECTANGLE_ASPECT_RATIO_MAX: shape_aspect_ok = True
                    elif shape in ["circle", "octagon"] and CIRCLE_OCTAGON_ASPECT_RATIO_MIN <= aspect_ratio <= CIRCLE_OCTAGON_ASPECT_RATIO_MAX: shape_aspect_ok = True

                    if shape_aspect_ok:
                        # <<< Bỏ comment nếu muốn lọc vị trí >>>
                        # if (y + h) < frame_height * MAX_VERTICAL_POSITION_RATIO: # Ví dụ lọc 65% trên
                        #     pass
                        # else: continue

                        # Cắt ROI
                        padding = 5
                        y1, y2 = max(0, y - padding), min(frame_height, y + h + padding)
                        x1, x2 = max(0, x - padding), min(frame_width, x + w + padding)
                        sign_roi = frame[y1:y2, x1:x2]

                        if sign_roi.size > 0:
                            # Phân loại
                            label_index, confidence = predict(sign_roi)

                            # Lọc Confidence
                            if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                                # Xử lý trùng lặp
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
                                        else: is_duplicate = True
                                        break
                                if not is_duplicate:
                                    label_text = labelToText.get(label_index, f"L:{label_index}")
                                    # Lưu box và label
                                    detected_signs_info[(center_x, center_y)] = {
                                        'box': [x, y, w, h], # Lưu list [x,y,w,h]
                                        'label': label_text,
                                        'confidence': confidence
                                    }

    # Chuyển đổi dict thành list rois và labels để trả về
    for sign_info in detected_signs_info.values():
        detected_rois.append(np.array(sign_info['box'])) # Chuyển thành numpy array
        detected_labels.append(sign_info['label'])

    return detected_rois, detected_labels

# --- Main Video Processing Loop ---
VIDEO_PATH = "static/videos/video32.mp4" # Đảm bảo đúng đường dẫn
cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video '{VIDEO_PATH}'.")
    exit()
print(f"Opened video: '{VIDEO_PATH}'")

isTracking = False # Bắt đầu không tracking
frame_count = 0
max_trackingFrame = 10 # Số frame tracking trước khi detect lại (có thể tăng/giảm)
trackers = None
current_labels = [] # Đổi tên `labels` thành `current_labels` để rõ ràng hơn

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # --- Logic Phát hiện hoặc Tracking ---
    if not isTracking:
        print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Detecting objects...")
        # Gọi hàm phát hiện MỚI
        rois, detected_labels_this_frame = findSigns(frame)
        print(f"Detected {len(rois)} signs passing filters.")

        if rois:
            trackers = cv.legacy.MultiTracker_create()
            current_labels = [] # Reset label cho các tracker mới
            valid_tracker_count = 0
            for i, roi in enumerate(rois):
                try:
                    # Chuyển ROI thành tuple số nguyên
                    if isinstance(roi, np.ndarray): roi_tuple = tuple(roi.astype(int))
                    elif isinstance(roi, (list, tuple)): roi_tuple = tuple(map(int, roi))
                    else: continue # Bỏ qua nếu kiểu không đúng

                    if roi_tuple[2] > 0 and roi_tuple[3] > 0: # w và h phải > 0
                        tracker_instance = cv.legacy.TrackerCSRT_create() # Hoặc thử KCF, MOSSE
                        # Thêm tracker vào MultiTracker
                        success_add = trackers.add(tracker_instance, frame, roi_tuple)
                        if success_add:
                             current_labels.append(detected_labels_this_frame[i]) # Chỉ thêm label nếu add tracker thành công
                             valid_tracker_count += 1
                        # else:
                        #     print(f"Warning: Failed to add tracker for ROI {roi_tuple}")
                    # else:
                    #     print(f"Warning: Invalid ROI dimensions {roi_tuple}. Skipping.")
                except Exception as e:
                    print(f"Error adding tracker for ROI {roi}: {e}")

            print(f"Initialized {valid_tracker_count} trackers.")
            if valid_tracker_count > 0:
                isTracking = True
                frame_count = 0
            else: # Không khởi tạo được tracker nào
                 isTracking = False
                 current_labels = [] # Reset labels

        else: # Không detect được ROI nào
            isTracking = False

    else: # Đang tracking
        # Kiểm tra nếu cần reset tracker
        if frame_count >= max_trackingFrame or trackers is None or len(trackers.getObjects()) == 0:
            print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Resetting tracker (timeout or lost).")
            isTracking = False; frame_count = 0; trackers = None; current_labels = []
            continue # Bỏ qua frame này, detect lại ở frame sau

        # Cập nhật tracker
        ret_track, objs = trackers.update(frame)

        # Kiểm tra số lượng tracker và label có khớp không
        if len(objs) != len(current_labels):
             print(f"Warning: Tracker count ({len(objs)}) mismatch with label count ({len(current_labels)}). Resetting.")
             isTracking = False; trackers = None; current_labels = []
             continue

        if ret_track:
            # Vẽ kết quả tracking
            for i, obj in enumerate(objs):
                p1 = (int(obj[0]), int(obj[1]))
                p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
                # Vẽ nếu tọa độ hợp lệ và kích thước > 0
                if p1[0] >= 0 and p1[1] >= 0 and p2[0] <= frame.shape[1] and p2[1] <= frame.shape[0] and obj[2]>0 and obj[3]>0:
                     cv.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                     label_to_draw = current_labels[i] # Lấy label tương ứng
                     (text_width, text_height), baseline = cv.getTextSize(label_to_draw, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                     # Đảm bảo nền text không vẽ ra ngoài ảnh
                     text_bg_y1 = max(0, p1[1] - text_height - baseline - 2)
                     text_y = max(text_height + baseline, p1[1] - baseline) # Đảm bảo text không vẽ ra ngoài
                     cv.rectangle(frame, (p1[0], text_bg_y1), (p1[0] + text_width + 2, p1[1]), (0,0,0), -1)
                     cv.putText(frame, label_to_draw, (p1[0] + 1, text_y -1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: Tracking update failed. Resetting.")
            isTracking = False; trackers = None; current_labels = []

        frame_count += 1

    # --- Hiển thị Frame ---
    cv.imshow('Video Sign Detection', frame)
    key = cv.waitKey(10) # Đợi 10ms, có thể giảm xuống 1 nếu muốn nhanh hơn
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('d'): # Nhấn 'd' để buộc detect lại
        print("Forcing re-detection...")
        isTracking = False; trackers = None; current_labels = []

# --- Dọn dẹp ---
cap.release()
cv.destroyAllWindows()
print("Video released and windows closed.")