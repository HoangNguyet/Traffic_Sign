import cv2 as cv
import numpy as np
from keras.models import load_model
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk
from threading import Thread

# Global font and color settings
font_face = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 1
text_color = (255, 255, 255)  # White (BGR)
bg_color = (0, 0, 255)        # Red (BGR)
text_padding = 5

# Load mô hình
try:
    model = load_model("Traffic_Sign/model_26.h5")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

# Định nghĩa nhãn GTSRB
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

def adjust_gamma(image, gamma=1.0):
    # Tăng cường độ sáng
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def returnHSV(img):
    # Tăng cường độ sáng trước khi chuyển đổi
    img = adjust_gamma(img, gamma=1.2)
    blur = cv.GaussianBlur(img, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Điều chỉnh ngưỡng màu cho GTSRB
RED_LOW1, RED_HIGH1 = (0, 120, 70), (10, 255, 255)
RED_LOW2, RED_HIGH2 = (170, 120, 70), (180, 255, 255)
BLUE_LOW, BLUE_HIGH = (100, 150, 50), (140, 255, 200)
YELLOW_LOW, YELLOW_HIGH = (20, 100, 100), (30, 255, 255)

def binaryImg(img):
    hsv = returnHSV(img)
    mask_red1 = cv.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask_red2 = cv.inRange(hsv, RED_LOW2, RED_HIGH2)
    b_img_red = cv.bitwise_or(mask_red1, mask_red2)
    b_img_blue = cv.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    b_img_yellow = cv.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    
    kernel = np.ones((3,3), np.uint8)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_CLOSE, kernel, iterations=2)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_OPEN, kernel, iterations=1)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_CLOSE, kernel, iterations=2)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_OPEN, kernel, iterations=1)
    b_img_yellow = cv.morphologyEx(b_img_yellow, cv.MORPH_CLOSE, kernel, iterations=2)
    b_img_yellow = cv.morphologyEx(b_img_yellow, cv.MORPH_OPEN, kernel, iterations=1)
    
    return b_img_red, b_img_blue, b_img_yellow

def preprocessing(img_roi):
    try:
        img_resized = cv.resize(img_roi, (32, 32))
        gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        equalized = cv.equalizeHist(gray)
        blurred = cv.GaussianBlur(equalized, (3,3), 0)
        img_processed = blurred / 255.0
        return img_processed
    except cv.error as e:
        print(f"Preprocessing error: {e}")
        return None

def predict(sign_roi):
    if sign_roi is None or sign_roi.size == 0: 
        return -1, 0.0
    
    img_processed = preprocessing(sign_roi)
    if img_processed is None: 
        return -1, 0.0
    
    try:
        img_reshaped = img_processed.reshape(1, 32, 32, 1)
        prediction_probabilities = model.predict(img_reshaped, verbose=0)
        predicted_class_index = np.argmax(prediction_probabilities)
        confidence = prediction_probabilities[0][predicted_class_index]
        
        MIN_CONFIDENCE = 0.6
        
        if confidence < MIN_CONFIDENCE: 
            return -1, confidence
            
        return predicted_class_index, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return -1, 0.0

def get_shape_type(contour, epsilon_factor=0.04):
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0: return None, 0, False, 0.0
    area = cv.contourArea(contour)
    if area < 10: return None, 0, False, 0.0
    
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0.0
    approx = cv.approxPolyDP(contour, epsilon_factor * perimeter, True)
    num_vertices = len(approx)
    is_convex = cv.isContourConvex(approx)
    
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0
    
    if num_vertices == 8 and circularity > 0.65:
        return "octagon", num_vertices, is_convex, circularity
    if num_vertices == 3 and circularity > 0.45:
        return "triangle", num_vertices, is_convex, circularity
    if num_vertices == 4 and 0.7 < aspect_ratio < 1.5 and circularity > 0.75:
        return "rectangle", num_vertices, is_convex, circularity
    if circularity > 0.8:
        return "circle", num_vertices, is_convex, circularity
    return None, num_vertices, is_convex, circularity

def findSigns(frame):
    if frame is None: return None
    
    # Resize để tăng tốc độ xử lý
    orig_height, orig_width = frame.shape[:2]
    scale_factor = 600 / max(orig_height, orig_width)
    if scale_factor < 1:
        frame = cv.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
    
    # Cân bằng sáng toàn ảnh
    frame = cv.convertScaleAbs(frame, alpha=1.2, beta=20)
    
    frame_height, frame_width = frame.shape[:2]
    b_img_red, b_img_blue, b_img_yellow = binaryImg(frame)
    
    contours_red, _ = cv.findContours(b_img_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv.findContours(b_img_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv.findContours(b_img_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    all_contours = []
    for c in contours_red: all_contours.append({'contour': c, 'color': 'red'})
    for c in contours_blue: all_contours.append({'contour': c, 'color': 'blue'})
    for c in contours_yellow: all_contours.append({'contour': c, 'color': 'yellow'})

    MIN_AREA = 300
    MIN_WIDTH = 20
    MIN_HEIGHT = 20
    IOU_THRESHOLD = 0.3

    detected_signs_info = []
    for item in all_contours:
        c = item['contour']
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)

        if area < MIN_AREA or w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
            
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (0.5 < aspect_ratio < 2.0): continue
        
        shape_name, _, _, _ = get_shape_type(c)
        if shape_name not in ["circle", "triangle", "rectangle", "octagon"]:
            continue

        padding = 15
        y1, y2 = max(0, y - padding), min(frame_height, y + h + padding)
        x1, x2 = max(0, x - padding), min(frame_width, x + w + padding)
                      
        sign_roi = frame[y1:y2, x1:x2]
        if sign_roi.size == 0: continue
        
        label_id, confidence = predict(sign_roi)
        if label_id == -1: continue
            
        label_text = f"{labelToText.get(label_id, f'Unknown ({label_id})')} ({confidence:.2f})"
        detected_signs_info.append([x, y, x+w, y+h, label_text, area, confidence])

    # Sắp xếp theo confidence và area
    detected_signs_info = sorted(detected_signs_info, key=lambda x: (x[6], x[5]), reverse=True)
    
    final_detections = []
    for i in range(len(detected_signs_info)):
        current_box_info = detected_signs_info[i]
        is_suppressed = False
        
        for j in range(len(final_detections)):
            existing_box_info = final_detections[j]
            
            xA = max(current_box_info[0], existing_box_info[0])
            yA = max(current_box_info[1], existing_box_info[1])
            xB = min(current_box_info[2], existing_box_info[2])
            yB = min(current_box_info[3], existing_box_info[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0: continue
            
            current_box_area = (current_box_info[2] - current_box_info[0]) * (current_box_info[3] - current_box_info[1])
            existing_box_area = (existing_box_info[2] - existing_box_info[0]) * (existing_box_info[3] - existing_box_info[1])
            
            if current_box_area == 0 or existing_box_area == 0: continue
            
            iou = interArea / float(current_box_area + existing_box_area - interArea)
            if iou > IOU_THRESHOLD:
                is_suppressed = True
                break
                
        if not is_suppressed:
            final_detections.append(current_box_info)

    drawn_text_rects = []
    for x1_draw, y1_draw, x2_draw, y2_draw, label_text, _, _ in final_detections:
        cv.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 2)
        
        (text_width, text_height), baseline = cv.getTextSize(label_text, font_face, font_scale, font_thickness)
        
        # Tính toán vị trí text
        text_x = x1_draw
        text_y = y1_draw - 10 if y1_draw - 10 > 10 else y2_draw + text_height + 10
        
        # Đảm bảo text không vượt ra khỏi frame
        if text_y < text_height + 10:
            text_y = y2_draw + text_height + 10
        if text_y > frame_height - 10:
            text_y = y1_draw - 10
            
        # Vẽ nền text
        cv.rectangle(frame, 
                    (text_x, text_y - text_height - baseline - text_padding),
                    (text_x + text_width + text_padding*2, text_y + baseline + text_padding),
                    bg_color, -1)
        
        # Vẽ text
        cv.putText(frame, label_text, 
                  (text_x + text_padding, text_y - text_padding), 
                  font_face, font_scale, text_color, font_thickness, cv.LINE_AA)
    
    # Resize lại kích thước ban đầu nếu cần
    if scale_factor < 1:
        frame = cv.resize(frame, (orig_width, orig_height))
        
    return frame

class TrafficSignApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Traffic Sign Detection - GTSRB")
        self.root.geometry("900x750")
        
        # Tạo theme đơn giản
        self.root.configure(bg="#f0f0f0")
        
        self.info_label = Label(self.root, text="Chọn ảnh để nhận diện biển báo GTSRB", 
                              font=("Arial", 14), bg="#f0f0f0", fg="#333")
        self.info_label.pack(pady=15)

        self.upload_button = Button(self.root, text="Chọn Ảnh", command=self.load_image, 
                                  font=("Arial", 12), width=20, bg="#4CAF50", fg="white")
        self.upload_button.pack(pady=10)

        self.panel_frame = tk.Frame(self.root, bg="lightgray", bd=2, relief=tk.SUNKEN)
        self.panel_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.image_panel = Label(self.panel_frame, bg="gray")
        self.image_panel.pack(fill=tk.BOTH, expand=True)
        
        self.current_image_cv = None
        
        # Thêm label hiển thị trạng thái
        self.status_label = Label(self.root, text="Sẵn sàng", font=("Arial", 10), 
                                bg="#f0f0f0", fg="#555")
        self.status_label.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        self.status_label.config(text="Đang xử lý...")
        self.upload_button.config(state=tk.DISABLED)
        
        def process_image():
            try:
                self.current_image_cv = cv.imread(file_path)
                if self.current_image_cv is None:
                    self.root.after(0, self.show_error, f"Không thể đọc ảnh từ {file_path.split('/')[-1]}")
                    return

                image_to_process = self.current_image_cv.copy()
                detected_image_cv = findSigns(image_to_process)
                
                if detected_image_cv is None:
                    self.root.after(0, self.show_error, "Lỗi trong quá trình xử lý ảnh.")
                    return

                self.root.after(0, self.update_gui, detected_image_cv, file_path)
                
            except Exception as e:
                self.root.after(0, self.show_error, str(e))
            finally:
                self.root.after(0, lambda: self.upload_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_label.config(text="Hoàn thành"))

        Thread(target=process_image, daemon=True).start()

    def update_gui(self, image_cv, file_path):
        detected_image_pil = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(detected_image_pil)

        panel_width = self.panel_frame.winfo_width()
        panel_height = self.panel_frame.winfo_height()

        if panel_width <= 1: panel_width = 800
        if panel_height <= 1: panel_height = 600

        img_pil.thumbnail((panel_width - 20, panel_height - 20), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_panel.config(image=img_tk)
        self.image_panel.image = img_tk
        self.info_label.config(text=f"Đã xử lý: {file_path.split('/')[-1]}")

    def show_error(self, error_msg):
        self.info_label.config(text=f"Lỗi: {error_msg}")
        self.status_label.config(text="Lỗi xảy ra")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()