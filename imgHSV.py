import cv2 as cv
import numpy as np
from keras.models import load_model # type: ignore
from tkinter import Tk, filedialog
from PIL import Image
from tkinter import Tk, filedialog, Button, Label
from PIL import Image, ImageTk
import tkinter as tk

CONFFIDENCE = 0.0
#Load mô hình
model = load_model("Traffic_Sign\model_24.h5")
 
#Định nghĩa các biển báo
labeltotext = {
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
    48:'Dont go Left', 49:'Dont go right', 50:'No Unturn', 51: 'No car', 52: 'No horn', 
    53:'Watch out for cars', 54: 'Horn', 55:'Uturn', 56:'Zebra Crossing', 57:'Fences', 58:'No Stopping'
    # Thêm các lớp khác nếu cần...
}

#Ngưỡng để lọc màu
low_thresh1, high_thresh1 = (165, 100, 40), (179, 255, 255)
low_thresh2, high_thresh2 = (0, 160, 40), (10, 255, 255)
low_thresh3, high_thresh3 = (100, 150, 40), (130, 255, 255)
low_thresh4, high_thresh4 = (25, 100, 100), (35, 255, 255)

#Chuyển ảnh sang HSV
def returnHSV(img):
    blur = cv.GaussianBlur(img, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

#Tạo ảnh nhị phân dựa trên màu sắc HSV
def binaryImg(img):
    hsv = returnHSV(img)
    b_img_red = cv.inRange(hsv, low_thresh1, high_thresh1) | cv.inRange(hsv, low_thresh2, high_thresh2)
    b_img_blue = cv.inRange(hsv, low_thresh3, high_thresh3)
    b_img_yellow = cv.inRange(hsv, low_thresh4, high_thresh4)

    #Giảm nhiễu
    kernel = np.ones((5,5), np.uint8)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_CLOSE, kernel)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_OPEN, kernel)

    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_CLOSE, kernel)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_OPEN, kernel)

    b_img_yellow = cv.morphologyEx(b_img_blue, cv.MORPH_CLOSE, kernel)
    b_img_yellow = cv.morphologyEx(b_img_blue, cv.MORPH_OPEN, kernel)

    return b_img_red, b_img_blue, b_img_yellow

#Xử lý ảnh trước khi đưa vào mô hình nhận diện
def grayscale(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv.equalizeHist(img)
    return img

def preprocessing(img):
    gray = grayscale(img)
    equalized = equalize(gray)
    img = equalized / 255
    return img

#Dự đoán nhãn biển báo
def predict(sign):
    img = preprocessing(sign)
    img = cv.resize(img, (32,32))
    img = img.reshape(1,32,32,1)
    predict_label = np.argmax(model.predict(img))
    confidence = np.max(model.predict(img))
    return predict_label, confidence
#Kiểm tra xem contour có dạng hình tròn không
def is_circular(contour):
    #Tính chu vi
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv.contourArea(contour) #diện tích
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 < circularity < 1.2 #Độ trong chấp nhận được

#Kiểm tra hình tam giác hoặc chữ nhật
def is_triangle_or_rectangle(contour):
    approx = cv.approxPolyDP(contour, 0.04*cv.arcLength(contour, True), True)
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return 'rectangle'
    return None

#Kiểm tra xem phải hình thoi hoặc bát giác không
def is_specific_shape(contour):
    approx = cv.approxPolyDP(contour, 0.04*cv.arcLength(contour, True), True)
    num_vertices = len(approx)
    if num_vertices == 8:
        return 'octagon'
    return None

def findSigns(frame):
    b_img_red, b_img_blue, b_img_yellow = binaryImg(frame)
    contour_red, _ = cv.findContours(b_img_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_blue, _ = cv.findContours(b_img_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_yellow, _ = cv.findContours(b_img_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detected_signs_info = {}
    all_contour = contour_red + contour_blue + contour_yellow
    used_text_positions = []  # Dùng để kiểm soát vị trí vẽ text, tránh bị đè lên nhau

    for c in all_contour:
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)

        shape = is_triangle_or_rectangle(c)
        is_circle = is_circular(c)
        specific_shape = is_specific_shape(c)

        # Điều kiện hình chữ nhật phải hợp lý (không quá dẹt)
        is_good_rectangle = shape == 'rectangle' and 0.5 <= w / h <= 1.5

        if area > 150 and (is_circle or shape == 'triangle' or is_good_rectangle or specific_shape == 'octagon'):
            padding = 15
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)
            sign_roi = frame[y1:y2, x1:x2]

            if sign_roi.size > 0:
                label_index, confidence = predict(sign_roi)
                if label_index != -1 and confidence >= CONFFIDENCE:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    is_duplicate = False

                    for center_key in list(detected_signs_info.keys()):
                        dist_sq = (center_x - center_key[0]) ** 2 + (center_y - center_key[1]) ** 2
                        old_w = detected_signs_info[center_key]['box'][2]
                        old_h = detected_signs_info[center_key]['box'][3]
                        #Ngưỡng để loại trừ contour trùng
                        threshold_dist_sq = ((max(w, old_w) / 2) ** 2 + (max(h, old_h) / 2) ** 2)

                        if dist_sq < threshold_dist_sq:
                            if (label_index == detected_signs_info[center_key]['label_index']):
                                if confidence > detected_signs_info[center_key]['confidence']:
                                    del detected_signs_info[center_key]
                                else:
                                    is_duplicate = True
                                break

                    if not is_duplicate:
                        label_text = labeltotext.get(label_index, f"U:{label_index}")
                        display_text = f"{label_text} ({confidence:.2f})"

                        # Tránh text bị vẽ đè lên nhau
                        text_x = max(0, min(x, frame.shape[1] - 150))
                        text_y = max(15, y - 10)
                        original_y = text_y

                        while any(abs(text_y - used_y) < 20 for used_y in used_text_positions):
                            text_y += 20  # Dời xuống nếu quá gần vị trí đã vẽ
                            if text_y > frame.shape[0] - 10:  # Nếu vượt khỏi frame, quay lại vị trí gốc
                                text_y = original_y
                                break
                        used_text_positions.append(text_y)

                        # Vẽ bounding box và text
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv.putText(frame, display_text, (text_x, text_y),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        detected_signs_info[(center_x, center_y)] = {
                            'box': (x, y, w, h),
                            'text': display_text,
                            'confidence': confidence
                        }

    return frame



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




