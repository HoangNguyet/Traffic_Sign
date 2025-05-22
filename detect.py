import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk

# Global font and color settings
font_face = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (255, 255, 255)  # White (BGR)
bg_color = (255, 0, 0)      # Blue (BGR)
text_padding = 3

# Load mô hình
try:
    model = load_model("Traffic_Sign/model_26.h5")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()
# print(model.summary()) # Có thể comment dòng này sau khi đã xem

# Định nghĩa nhãn
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
    42: 'End no passing vehicles > 3.5 tons', 43:'Speed limit (5km/h)', 44: 'Speed limit (15km/h)',
    45: 'Speed limit (40km/h)', 46: 'Dont go straight or left', 47:'Dont go straight',
    48:'Dont go Left', 49:'Dont go right', 50:'No Uturn', 51: 'No car', 52: 'No horn',
    53:'Watch out for cars', 54: 'Horn', 55:'Uturn', 56:'Zebra Crossing', 57:'Fences', 58:'No Stopping'
}

def returnHSV(img):
    blur = cv.GaussianBlur(img, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

RED_LOW1, RED_HIGH1 = (0, 160, 40), (10, 255, 255)
RED_LOW2, RED_HIGH2 = (165, 100, 40), (179, 255, 255)
BLUE_LOW, BLUE_HIGH = (100, 150, 40), (130, 255, 255)
# YELLOW_LOW, YELLOW_HIGH = (20, 80, 80), (30, 255, 255)

def binaryImg(img):
    hsv = returnHSV(img)
    mask_red1 = cv.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask_red2 = cv.inRange(hsv, RED_LOW2, RED_HIGH2)
    b_img_red = cv.bitwise_or(mask_red1, mask_red2)
    b_img_blue = cv.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    # b_img_yellow = cv.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    kernel = np.ones((3,3), np.uint8) # Sử dụng kernel nhỏ hơn
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_CLOSE, kernel, iterations=2)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_OPEN, kernel, iterations=1)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_CLOSE, kernel, iterations=2)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_OPEN, kernel, iterations=1)
    # b_img_yellow = cv.morphologyEx(b_img_yellow, cv.MORPH_CLOSE, kernel, iterations=2)
    # b_img_yellow = cv.morphologyEx(b_img_yellow, cv.MORPH_OPEN, kernel, iterations=1)
    # Debug
    cv.imshow("Red Binary", b_img_red)
    cv.imshow("Blue Binary", b_img_blue)
    # cv.imshow("Yellow Binary", b_img_yellow)
    cv.waitKey(1)
    return b_img_red, b_img_blue

def preprocessing(img_roi):
    try:
        gray = cv.cvtColor(img_roi, cv.COLOR_BGR2GRAY)
        equalized = cv.equalizeHist(gray)
        #Giảm nhiễu
        # blur = cv.GaussianBlur(equalized, (3,3), 0)
        img_processed = equalized / 255.0
        return img_processed
    except cv.error:
        return None

def predict(sign_roi):
    if sign_roi is None or sign_roi.size == 0: return -1
    img_processed = preprocessing(sign_roi)
    if img_processed is None: return -1
    try:
        img_resized = cv.resize(img_processed, (32, 32))
        img_reshaped = img_resized.reshape(1, 32, 32, 1)
        prediction_probabilities = model.predict(img_reshaped, verbose=0)
        predicted_class_index = np.argmax(prediction_probabilities)
        confidence = prediction_probabilities[0][predicted_class_index]
        MIN_CONFIDENCE = 0.2
        if confidence < MIN_CONFIDENCE: return -1
        return predicted_class_index
    except Exception:
        return -1

def get_shape_type(contour, epsilon_factor=0.03, min_circularity_for_octagon=0.5, min_circularity_for_circle=0.5, min_circularity_general=0.6):
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0: return None, 0, False, 0.0
    area = cv.contourArea(contour)
    if area < 10: return None, 0, False, 0.0
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0.0
    approx = cv.approxPolyDP(contour, epsilon_factor * perimeter, True)
    num_vertices = len(approx)
    is_convex = cv.isContourConvex(approx)
    if num_vertices == 8 and is_convex and circularity > min_circularity_for_octagon:
        return "octagon", num_vertices, is_convex, circularity
    if is_convex and circularity > min_circularity_for_circle:
        return "circle", num_vertices, is_convex, circularity
    if num_vertices == 3 and is_convex:
        return "triangle", num_vertices, is_convex, circularity
    if num_vertices == 4 and is_convex:
        return "rectangle", num_vertices, is_convex, circularity
    if is_convex and circularity > min_circularity_general and num_vertices > 4:
        return "circle_ish", num_vertices, is_convex, circularity
    return None, num_vertices, is_convex, circularity

def findSigns(frame):
    if frame is None: return None
    frame_height, frame_width = frame.shape[:2]
    b_img_red, b_img_blue = binaryImg(frame)
    contours_red, _ = cv.findContours(b_img_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv.findContours(b_img_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours_yellow, _ = cv.findContours(b_img_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    all_contours = []
    for c in contours_red: all_contours.append({'contour': c, 'color': 'red'})
    for c in contours_blue: all_contours.append({'contour': c, 'color': 'blue'})
    # for c in contours_yellow: all_contours.append({'contour': c, 'color': 'yellow'})

    MIN_AREA = 200
    MIN_WIDTH = 5
    MIN_HEIGHT = 5
    IOU_THRESHOLD = 0.3
    MIN_SOLIDITY = 0.55
    MIN_EXTENT = 0.2

    detected_signs_info = []
    for item in all_contours:
        c = item['contour']
        area = cv.contourArea(c)
        x,y,w,h = cv.boundingRect(c)

        if area < MIN_AREA:
            continue
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (0.5 < aspect_ratio < 2.0): continue
        
        # hull = cv.convexHull(c)
        # hull_area = cv.contourArea(hull)
        # if hull_area == 0: continue
        # solidity = float(area) / hull_area
        # if solidity < MIN_SOLIDITY: continue

        # rect_area = w * h
        # if rect_area == 0: continue
        # extent = float(area) / rect_area
        # if extent < MIN_EXTENT: continue

        # Bỏ qua kiểm tra hình dạng từ get_shape_type nếu bạn muốn
        # shape_name, _, _, _ = get_shape_type(c)
        # valid_shapes = ["circle", "triangle", "rectangle", "octagon", "circle_ish"]
        # if shape_name not in valid_shapes:
        #     continue
        padding = 15
        y1, y2 = max(0, y - padding), min(frame.shape[0], y + h + padding)
        x1, x2 = max(0, x - padding), min(frame.shape[1], x + w + padding)
                      

        # if roi_x_end > roi_x_start and roi_y_end > roi_y_start:
        sign_roi = frame[y1:y2, x1:x2]
        if sign_roi.size == 0: continue
        label_id = predict(sign_roi)
        if label_id == -1: continue
        label_text = labelToText.get(label_id, f"Unknown ({label_id})")
        detected_signs_info.append([x, y, x+w, y+h, label_text, area])

    detected_signs_info = sorted(detected_signs_info, key=lambda x: x[5], reverse=True)
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
            if current_box_area == 0 or existing_box_area == 0 : continue
            iou = interArea / float(current_box_area + existing_box_area - interArea)
            if iou > IOU_THRESHOLD:
                is_suppressed = True
                break
        if not is_suppressed:
            final_detections.append(current_box_info)

    # --- PHẦN VẼ TEXT ĐƯỢC CẬP NHẬT ---
    drawn_text_rects = [] # Lưu trữ các hình chữ nhật của nền text đã vẽ

    for x1_draw, y1_draw, x2_draw, y2_draw, label_text, _ in final_detections:
        cv.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 2)
        
        (text_width, text_height), baseline = cv.getTextSize(label_text, font_face, font_scale, font_thickness)
        
        # Tính toán vị trí ban đầu cho nền text (phía trên bounding box)
        current_bg_y_start = y1_draw - text_height - baseline - text_padding * 2
        
        # Vị trí X của nền text
        current_bg_x_start = x1_draw
        current_bg_x_end = current_bg_x_start + text_width + text_padding * 2
        current_bg_y_end = current_bg_y_start + text_height + baseline + text_padding * 2

        # Điều chỉnh để tránh chồng lấn và tràn lề
        max_attempts = 10 # Số lần thử dịch chuyển tối đa
        attempt = 0
        overlap = True

        # Ưu tiên đặt phía trên, sau đó phía dưới, rồi dịch chuyển
        placement_options = ["above", "below"] 
        
        final_bg_x_start, final_bg_y_start = -1, -1
        final_bg_x_end, final_bg_y_end = -1, -1

        for placement in placement_options:
            if not overlap: break # Nếu đã tìm được vị trí không chồng lấn

            if placement == "above":
                current_bg_y_start = y1_draw - text_height - baseline - text_padding * 2
            else: # placement == "below"
                current_bg_y_start = y2_draw + text_padding

            attempt = 0
            while attempt < max_attempts and overlap:
                current_bg_x_start = x1_draw # Giữ nguyên x_start ban đầu cho mỗi lần thử placement
                current_bg_y_end = current_bg_y_start + text_height + baseline + text_padding * 2
                current_bg_x_end = current_bg_x_start + text_width + text_padding * 2

                # Giới hạn trong khung hình (quan trọng)
                temp_bg_x_start = max(0, current_bg_x_start)
                temp_bg_y_start = max(0, current_bg_y_start)
                temp_bg_x_end = min(frame_width, current_bg_x_end)
                temp_bg_y_end = min(frame_height, current_bg_y_end)
                
                # Chỉ kiểm tra chồng lấn nếu kích thước nền hợp lệ
                if temp_bg_x_end <= temp_bg_x_start or temp_bg_y_end <= temp_bg_y_start:
                    # Không thể vẽ ở vị trí này, thử dịch chuyển tiếp
                    if placement == "above":
                         current_bg_y_start -= (text_height // 2) # Dịch lên chút
                    else: # below
                         current_bg_y_start += (text_height // 2) # Dịch xuống chút
                    attempt += 1
                    continue

                current_text_rect = (temp_bg_x_start, temp_bg_y_start, temp_bg_x_end, temp_bg_y_end)
                overlap = False # Giả sử không chồng lấn
                for r_idx, drawn_rect in enumerate(drawn_text_rects):
                    # Kiểm tra chồng lấn (IoU đơn giản hoặc kiểm tra giao nhau)
                    # (x1_a, y1_a, x2_a, y2_a) = current_text_rect
                    # (x1_b, y1_b, x2_b, y2_b) = drawn_rect
                    # is_overlapping = not (x2_a < x1_b or x1_a > x2_b or y2_a < y1_b or y1_a > y2_b)

                    # Kiểm tra chồng lấn bằng cách xem khoảng cách giữa các cạnh
                    # Chồng lấn nếu khoảng cách theo chiều y nhỏ hơn chiều cao của text
                    # và chúng nằm trên cùng một cột x (có thể nới lỏng điều kiện x)
                    dist_y = abs((current_text_rect[1] + current_text_rect[3]) / 2 - (drawn_rect[1] + drawn_rect[3]) / 2)
                    min_dist_y = ( (current_text_rect[3]-current_text_rect[1]) + (drawn_rect[3]-drawn_rect[1]) ) / 2
                    
                    # Kiểm tra chồng lấn theo chiều X
                    overlap_x = not (current_text_rect[2] < drawn_rect[0] or current_text_rect[0] > drawn_rect[2])

                    if dist_y < min_dist_y and overlap_x:
                        overlap = True
                        break
                
                if not overlap:
                    final_bg_x_start, final_bg_y_start = temp_bg_x_start, temp_bg_y_start
                    final_bg_x_end, final_bg_y_end = temp_bg_x_end, temp_bg_y_end
                    break # Tìm được vị trí tốt

                # Nếu chồng lấn, thử dịch chuyển (cho cả "above" và "below")
                if placement == "above":
                    current_bg_y_start -= (text_height // 2 + text_padding) # Dịch lên
                else: # below
                    current_bg_y_start += (text_height // 2 + text_padding) # Dịch xuống
                attempt += 1
            
        # Nếu sau tất cả các lần thử vẫn không tìm được vị trí hoàn hảo,
        # sử dụng vị trí cuối cùng đã tính toán (có thể vẫn bị chồng lấn hoặc tràn)
        # hoặc vị trí ban đầu nếu không có vị trí nào tốt hơn
        if final_bg_x_start == -1 : # Không tìm được vị trí không chồng lấn nào
             # Dùng vị trí ban đầu (phía trên) và giới hạn nó
            final_bg_y_start = y1_draw - text_height - baseline - text_padding * 2
            final_bg_x_start = x1_draw
            final_bg_y_end = final_bg_y_start + text_height + baseline + text_padding * 2
            final_bg_x_end = final_bg_x_start + text_width + text_padding * 2

            final_bg_x_start = max(0, final_bg_x_start)
            final_bg_y_start = max(0, final_bg_y_start)
            final_bg_x_end = min(frame_width, final_bg_x_end)
            final_bg_y_end = min(frame_height, final_bg_y_end)


        # Vẽ nền và text tại vị trí cuối cùng đã chọn
        if final_bg_x_end > final_bg_x_start and final_bg_y_end > final_bg_y_start:
            cv.rectangle(frame, (final_bg_x_start, final_bg_y_start), (final_bg_x_end, final_bg_y_end), bg_color, -1)
            
            text_origin_x = final_bg_x_start + text_padding
            text_origin_y = final_bg_y_start + text_height + text_padding
            cv.putText(frame, label_text, (text_origin_x, text_origin_y), 
                        font_face, font_scale, text_color, font_thickness, cv.LINE_AA)
            
            drawn_text_rects.append((final_bg_x_start, final_bg_y_start, final_bg_x_end, final_bg_y_end))
    # --- KẾT THÚC PHẦN VẼ TEXT ---
    return frame

# --- Lớp GUI giữ nguyên ---
class TrafficSignApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Traffic Sign Detection")
        self.root.geometry("800x700")

        self.info_label = Label(self.root, text="Chọn ảnh để nhận diện biển báo.", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.upload_button = Button(self.root, text="Chọn Ảnh", command=self.load_image, font=("Arial", 12), width=20)
        self.upload_button.pack(pady=10)

        self.panel_frame = tk.Frame(self.root, bg="lightgray", bd=2, relief=tk.SUNKEN)
        self.panel_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.image_panel = Label(self.panel_frame, bg="gray")
        self.image_panel.pack(fill=tk.BOTH, expand=True)
        
        self.current_image_cv = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.current_image_cv = cv.imread(file_path)
            if self.current_image_cv is None:
                self.info_label.config(text=f"Lỗi: Không thể đọc ảnh từ {file_path.split('/')[-1]}")
                return

            image_to_process = self.current_image_cv.copy()
            
            roi_padding = 15
            draw_padding = 0 # Padding cho bounding box của biển báo
            
            detected_image_cv = findSigns(image_to_process)
            
            if detected_image_cv is None:
                self.info_label.config(text="Lỗi trong quá trình xử lý ảnh.")
                return

            detected_image_pil = cv.cvtColor(detected_image_cv, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(detected_image_pil)

            self.root.update_idletasks() 
            panel_width = self.panel_frame.winfo_width()
            panel_height = self.panel_frame.winfo_height()

            if panel_width <= 1: panel_width = 700
            if panel_height <= 1: panel_height = 550

            img_pil.thumbnail((panel_width - 10, panel_height - 10), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img_pil)

            self.image_panel.config(image=img_tk)
            self.image_panel.image = img_tk
            self.info_label.config(text=f"Đã xử lý: {file_path.split('/')[-1]}")

        except Exception as e:
            print(f"Lỗi khi tải hoặc xử lý ảnh: {e}")
            self.info_label.config(text=f"Lỗi: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()