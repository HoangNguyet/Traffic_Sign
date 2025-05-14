import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk
import math

# --- Constants ---
MIN_SIGN_AREA = 150
MAX_SIGN_AREA = 80000
APPROX_EPSILON_FACTOR = 0.03
CIRCULARITY_THRESHOLD_FOR_CIRCLE = 0.2
SOLIDITY_MIN_THRESHOLD = 0.65
RECTANGLE_ASPECT_RATIO_MIN = 1.0
RECTANGLE_ASPECT_RATIO_MAX = 2.0
TRIANGLE_ASPECT_RATIO_MIN = 0.6
TRIANGLE_ASPECT_RATIO_MAX = 1.4
CIRCLE_OCTAGON_ASPECT_RATIO_MIN = 0.1
CIRCLE_OCTAGON_ASPECT_RATIO_MAX = 1.2
CONFIDENCE_THRESHOLD = 0.7  # Tăng ngưỡng tin cậy
RESIZE_DIM = (32, 32)

# --- Load Model and Labels ---
try:
    model = load_model("model_24.h5")
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
    39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicles > 3.5 tons'
}

# --- Image Processing Functions ---
def returnHSV(img):
    """Chuyển ảnh sang HSV và làm mờ nhẹ."""
    blur = cv.GaussianBlur(img, (5, 5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu HSV
low_thresh_red1, high_thresh_red1 = (165, 40, 40), (179, 255, 255)
low_thresh_red2, high_thresh_red2 = (0, 40, 40), (10, 255, 255)
low_thresh_blue, high_thresh_blue = (100, 150, 40), (130, 255, 255)
low_thresh_yellow, high_thresh_yellow = (15, 192, 147), (22, 255, 255)

def create_binary_mask_hsv(hsv_img):
    """Tạo mask nhị phân kết hợp cho các màu đỏ, xanh, vàng từ ảnh HSV."""
    mask_red1 = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1)
    mask_red2 = cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)
    
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)
    
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    return combined_mask

def preprocessing(img):
    """Tiền xử lý ảnh đầu vào cho mô hình CNN."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    resized = cv.resize(equalized, RESIZE_DIM, interpolation=cv.INTER_AREA)
    normalized = resized / 255.0
    return normalized

def predict(sign_image):
    """Dự đoán nhãn và độ tin cậy cho ảnh biển báo."""
    if sign_image is None or sign_image.size == 0:
        return -1, 0.0
    try:
        processed_img = preprocessing(sign_image)
        img_array = processed_img.reshape(1, RESIZE_DIM[0], RESIZE_DIM[1], 1)
        prediction = model.predict(img_array, verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1, 0.0

def identify_shape(contour):
    """Xác định hình dạng và tính các thuộc tính."""
    shape = "unknown"
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0: return shape, 0.0, 0.0

    area = cv.contourArea(contour)
    if area < MIN_SIGN_AREA / 2: return shape, 0.0, 0.0

    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

    epsilon = APPROX_EPSILON_FACTOR * perimeter
    approx = cv.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        shape = "rectangle"
    elif num_vertices == 8:
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE - 0.15:
             shape = "octagon"
    elif num_vertices >= 7:
        if circularity >= CIRCULARITY_THRESHOLD_FOR_CIRCLE:
             shape = "circle"

    return shape, solidity, circularity

def findSigns(frame):
    """Phát hiện và phân loại biển báo chỉ sử dụng màu sắc HSV."""
    output_frame = frame.copy()
    detected_signs_info = {}

    # Chỉ sử dụng xử lý HSV (đã bỏ phần grayscale)
    hsv = returnHSV(frame)
    binary_mask = create_binary_mask_hsv(hsv)
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv.contourArea(c)
        if MIN_SIGN_AREA < area: 
            shape, solidity, circularity = identify_shape(c)

            if solidity >= SOLIDITY_MIN_THRESHOLD and shape != "unknown":
                x, y, w, h = cv.boundingRect(c)
                aspect_ratio = float(w) / h if h > 0 else 0

                shape_aspect_ok = False
                if shape == "triangle" and TRIANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= TRIANGLE_ASPECT_RATIO_MAX:
                    shape_aspect_ok = True
                elif shape == "rectangle" and RECTANGLE_ASPECT_RATIO_MIN <= aspect_ratio <= RECTANGLE_ASPECT_RATIO_MAX:
                    shape_aspect_ok = True
                elif shape in ["circle", "octagon"] and CIRCLE_OCTAGON_ASPECT_RATIO_MIN <= aspect_ratio <= CIRCLE_OCTAGON_ASPECT_RATIO_MAX:
                    shape_aspect_ok = True

                if shape_aspect_ok:
                    padding = 5
                    y1, y2 = max(0, y - padding), min(frame.shape[0], y + h + padding)
                    x1, x2 = max(0, x - padding), min(frame.shape[1], x + w + padding)
                    sign_roi = frame[y1:y2, x1:x2]

                    if sign_roi.size > 0:
                        label_index, confidence = predict(sign_roi)

                        if label_index != -1 and confidence >= CONFIDENCE_THRESHOLD:
                            center_x, center_y = x + w // 2, y + h // 2
                            is_duplicate = False
                            for center_key in list(detected_signs_info.keys()):
                                dist_sq = (center_x - center_key[0])**2 + (center_y - center_key[1])**2
                                old_w = detected_signs_info[center_key]['box'][2]
                                old_h = detected_signs_info[center_key]['box'][3]
                                threshold_dist_sq = ((max(w, old_w)/4)**2 + (max(h, old_h)/4)**2)

                                if dist_sq < threshold_dist_sq:
                                    if confidence > detected_signs_info[center_key]['confidence']:
                                        del detected_signs_info[center_key]
                                    else:
                                        is_duplicate = True
                                    break

                            if not is_duplicate:
                                label_text = labelToText.get(label_index, f"U:{label_index}")
                                display_text = f"{label_text} ({confidence:.2f})"
                                detected_signs_info[(center_x, center_y)] = {
                                    'box': (x, y, w, h),
                                    'text': display_text,
                                    'y_pos': y,
                                    'confidence': confidence
                                }

    # Vẽ kết quả
    detected_signs = list(detected_signs_info.values())
    detected_signs.sort(key=lambda item: item['y_pos'])

    last_text_y = -100
    text_gap = 20

    for sign_info in detected_signs:
        x, y, w, h = sign_info['box']
        display_text = sign_info['text']

        (text_width, text_height), baseline = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_x = max(10, min(x, frame.shape[1] - text_width - 10))
        text_y = y - 10 if y - text_height - 10 > 10 else y + h + text_height + 10
        
        if text_y < last_text_y + text_gap:
            text_y = last_text_y + text_gap
        
        text_y = min(text_y, frame.shape[0] - text_height - 10)
        
        cv.rectangle(output_frame, 
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y + baseline),
                    (0, 0, 0), -1)
        
        cv.putText(output_frame, display_text, 
                  (text_x, text_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        last_text_y = text_y

    return output_frame

# --- GUI Class ---
class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection (Color Only)")
        self.root.geometry("1000x850")

        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Title label
        self.label = Label(root, text="Traffic Sign Detection (Color Detection Only)", 
                         font=("Arial", 16, "bold"), pady=10)
        self.label.grid(row=0, column=0, sticky="n")
        
        # Upload button
        self.upload_button = Button(root, text="Choose Image", command=self.load_image, 
                                  font=("Arial", 12), width=20, height=2)
        self.upload_button.grid(row=0, column=0, pady=10, sticky="s")
        
        # Image frame
        self.image_frame = tk.Frame(root, bg="lightgray", bd=2, relief="groove")
        self.image_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.image_frame.grid_propagate(False)
        
        # Image panel
        self.panel = Label(self.image_frame, bg="lightgray")
        self.panel.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Status bar
        self.status_label = Label(root, text="Ready to upload image", 
                                font=("Arial", 10), fg="gray", bd=1, relief="sunken")
        self.status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

    def load_image(self):
        file_types = [
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=file_types)
        if not file_path:
            self.status_label.config(text="Image selection cancelled.")
            return

        try:
            self.status_label.config(text=f"Processing: {file_path.split('/')[-1]}...")
            self.root.update_idletasks()

            image = cv.imread(file_path)
            if image is None:
                raise ValueError("Could not read image file")
                
            detected_image = findSigns(image)
            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)
            
            self.image_frame.update()
            frame_w = self.image_frame.winfo_width() - 20
            frame_h = self.image_frame.winfo_height() - 20
            
            img_w, img_h = pil_image.size
            ratio = min(frame_w/img_w, frame_h/img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))
            
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil_image)
            
            self.panel.config(image=imgtk)
            self.panel.image = imgtk
            
            self.status_label.config(text=f"Processed: {file_path.split('/')[-1]}")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()