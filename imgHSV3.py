import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk

# --- Constants ---
MIN_SIGN_AREA = 150      # Tăng ngưỡng diện tích tối thiểu
MAX_SIGN_AREA = 80000
CONFIDENCE_THRESHOLD = 0.5
RESIZE_DIM = (32, 32)

# --- Load Model and Labels ---
try:
    model = load_model("Traffic_Sign\model_24.h5")
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
    39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicles > 3.5 tons'
}

# --- Image Processing Functions ---
def returnHSV(img):
    blur = cv.GaussianBlur(img, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Ngưỡng màu cập nhật
low_thresh_red1, high_thresh_red1 = (165, 100, 40), (179, 255, 255)
low_thresh_red2, high_thresh_red2 = (0, 160, 40), (10, 255, 255)
low_thresh_blue, high_thresh_blue = (100, 150, 40), (130, 255, 255)
# low_thresh_yellow, high_thresh_yellow = (25, 100, 100), (35, 255, 255)

def create_binary_mask_hsv(hsv_img):
    mask_red = cv.inRange(hsv_img, low_thresh_red1, high_thresh_red1) | cv.inRange(hsv_img, low_thresh_red2, high_thresh_red2)
    mask_blue = cv.inRange(hsv_img, low_thresh_blue, high_thresh_blue)
    # mask_yellow = cv.inRange(hsv_img, low_thresh_yellow, high_thresh_yellow)
    
    kernel = np.ones((5,5), np.uint8)
    mask_red = cv.morphologyEx(mask_red, cv.MORPH_CLOSE, kernel)
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)
    # mask_yellow = cv.morphologyEx(mask_yellow, cv.MORPH_CLOSE, kernel)
    
    return mask_red, mask_blue

def preprocessing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    resized = cv.resize(equalized, RESIZE_DIM, interpolation=cv.INTER_AREA)
    normalized = resized / 255.0
    return normalized

def predict(sign_image):
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

def is_circular(contour):
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 < circularity < 1.2

def is_triangle_or_rectangle(contour):
    approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "rectangle"
    return None

def is_specific_shape(contour):
    approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    num_vertices = len(approx)
    if num_vertices == 4:
        return "diamond"
    elif num_vertices == 8:
        return "octagon"
    return None

def findSigns(frame):
    output_frame = frame.copy()
    hsv = returnHSV(frame)
    mask_red, mask_blue = create_binary_mask_hsv(hsv)
    
    contours_red, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    all_contours = contours_red + contours_blue
    used_positions = [] 
    
    # Define padding amount. This will be used for both ROI extraction and drawing the box.
    # Code 1 uses padding = 5 for its ROI.
    BOX_PADDING = 0

    for c in all_contours:
        area = cv.contourArea(c)
        # Get original bounding box coordinates from the contour
        x_orig, y_orig, w_orig, h_orig = cv.boundingRect(c)
        
        # Shape checking logic from original code 2
        shape_type = is_triangle_or_rectangle(c)
        is_circ = is_circular(c)
        specific_shape_type = is_specific_shape(c)
        
        if area > MIN_SIGN_AREA and \
           (is_circ or shape_type in ["triangle", "rectangle"] or specific_shape_type in ["diamond", "octagon"]):
            
            # --- ROI for Prediction (Padded) ---
            roi_y1 = max(0, y_orig - BOX_PADDING)
            roi_y2 = min(frame.shape[0], y_orig + h_orig + BOX_PADDING)
            roi_x1 = max(0, x_orig - BOX_PADDING)
            roi_x2 = min(frame.shape[1], x_orig + w_orig + BOX_PADDING)
            
            sign_roi_for_prediction = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            label, confidence = -1, 0.0
            if sign_roi_for_prediction.size > 0:
                label, confidence = predict(sign_roi_for_prediction)
            
            if confidence < CONFIDENCE_THRESHOLD:
                continue
                
            label_text = labelToText.get(label, f"Unknown: {label}")
            display_text = f"{label_text} ({confidence:.2f})"
            
            # --- Coordinates for Drawing the Bounding Box (Padded) ---
            draw_box_x1 = max(0, x_orig - BOX_PADDING)
            draw_box_y1 = max(0, y_orig - BOX_PADDING)
            draw_box_x2 = min(frame.shape[1], x_orig + w_orig + BOX_PADDING)
            draw_box_y2 = min(frame.shape[0], y_orig + h_orig + BOX_PADDING)

            # --- Text Position Calculation (relative to the padded drawn box) ---
            text_x = draw_box_x1
            # Place text above the padded box, or below if it's too close to the top
            text_y = draw_box_y1 - 10 
            if text_y < 15: # If text_y is too close to the top edge (e.g., < 15 pixels)
                text_y = draw_box_y2 + 20 # Place it below the padded box

            # Simple text overlap avoidance (from original code 2, adjusted for padded box)
            min_vertical_distance = 25 
            for (prev_text_x_stored, prev_text_y_stored) in used_positions:
                if abs(text_y - prev_text_y_stored) < min_vertical_distance:
                    text_y = max(text_y, prev_text_y_stored + min_vertical_distance) 
                    if text_y + 20 > frame.shape[0]: 
                        text_y = draw_box_y2 + 20 # Fallback if pushed too far
            
            used_positions.append((text_x, text_y))
            
            # --- Drawing ---
            # Draw the PADDED bounding box
            cv.rectangle(output_frame, (draw_box_x1, draw_box_y1), (draw_box_x2, draw_box_y2), (0, 255, 0), 2)
            
            (text_width, text_height), baseline = cv.getTextSize(
                display_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            bg_rect_y1 = text_y - text_height - 5 
            bg_rect_y2 = text_y + baseline + 5     
            cv.rectangle(output_frame,
                        (text_x, bg_rect_y1),
                        (text_x + text_width + 5, bg_rect_y2), 
                        (0, 0, 0), -1) 
            
            cv.putText(output_frame, display_text, (text_x, text_y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output_frame

class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection")
        self.root.geometry("1000x800")  # Tăng kích thước cửa sổ
        
        # Tạo frame chính với padding
        main_frame = tk.Frame(root, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")
        
        # Label tiêu đề
        self.label = Label(main_frame, 
                         text="Traffic Sign Recognition System",
                         font=("Arial", 16, "bold"))
        self.label.pack(pady=(0, 20))
        
        # Nút upload
        self.upload_button = Button(main_frame, 
                                   text="Choose Image", 
                                   command=self.load_image,
                                   font=("Arial", 12),
                                   width=20,
                                   height=2,
                                   bg="#4CAF50",
                                   fg="white")
        self.upload_button.pack(pady=10)
        
        # Frame chứa ảnh với scrollbar
        self.image_container = tk.Frame(main_frame)
        self.image_container.pack(expand=True, fill="both")
        
        # Canvas để hiển thị ảnh với scrollbar
        self.canvas = tk.Canvas(self.image_container, bg="lightgray")
        self.scroll_y = tk.Scrollbar(self.image_container, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.image_container, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", expand=True, fill="both")
        
        # Frame bên trong canvas để chứa ảnh
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        
        # Label hiển thị ảnh
        self.panel = Label(self.image_frame)
        self.panel.pack()
        
        # Label trạng thái
        self.status_label = Label(main_frame, 
                                text="Please upload an image to detect traffic signs.",
                                font=("Arial", 10),
                                fg="gray")
        self.status_label.pack(pady=(10, 0))
        
        # Cấu hình resize
        self.image_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig("all", width=event.width)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            self.status_label.config(text="Image selection cancelled.")
            return

        try:
            self.status_label.config(text=f"Processing: {file_path.split('/')[-1]}...")
            self.root.update_idletasks()

            image = cv.imread(file_path)
            if image is None:
                self.status_label.config(text=f"Error: Could not read image file.")
                return

            detected_image = findSigns(image)
            detected_image_rgb = cv.cvtColor(detected_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detected_image_rgb)

            # Tính toán tỷ lệ resize để vừa với canvas
            canvas_width = self.canvas.winfo_width() - 20
            canvas_height = self.canvas.winfo_height() - 20
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 800, 600
                
            ratio = min(canvas_width / pil_image.width, canvas_height / pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil_image)

            # Cập nhật ảnh
            self.panel.config(image=imgtk)
            self.panel.image = imgtk
            
            # Cập nhật scrollregion sau khi thêm ảnh mới
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            self.status_label.config(text=f"Detection complete: {file_path.split('/')[-1]}")

        except Exception as e:
            print(f"Error: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()