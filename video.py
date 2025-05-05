import cv2 as cv
import numpy as np
from keras.models import load_model  # type: ignore

# Load mô hình nhận diện biển báo
model = load_model("model1.h5")

# Định nghĩa nhãn biển báo
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

# Ngưỡng màu sắc
low_thresh1, high_thresh1 = (165, 100, 40), (179, 255, 255)  # Đỏ
low_thresh2, high_thresh2 = (0, 160, 40), (10, 255, 255)     # Đỏ
low_thresh3, high_thresh3 = (100, 150, 40), (130, 255, 255)  # Xanh dương

# Xử lý ảnh trước khi đưa vào mô hình
def preprocessing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    img = equalized / 255
    return img

# Dự đoán nhãn biển báo
def predict(sign):
    img = preprocessing(sign)
    img = cv.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 1)
    return np.argmax(model.predict(img))

# Chuyển đổi ảnh sang HSV và tạo ảnh nhị phân dựa trên màu sắc
def binaryImg(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    b_img_red = cv.inRange(hsv, low_thresh1, high_thresh1) | cv.inRange(hsv, low_thresh2, high_thresh2)
    b_img_blue = cv.inRange(hsv, low_thresh3, high_thresh3)
    
    # Giảm nhiễu bằng phép toán đóng (Closing)
    kernel = np.ones((5,5), np.uint8)
    b_img_red = cv.morphologyEx(b_img_red, cv.MORPH_CLOSE, kernel)
    b_img_blue = cv.morphologyEx(b_img_blue, cv.MORPH_CLOSE, kernel)
    return b_img_red, b_img_blue

# Kiểm tra hình dạng của biển báo
def is_circular(contour):
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.7 < circularity < 1.2  # Hình tròn có giá trị gần 1

def is_triangle_or_rectangle(contour):
    approx = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
    if len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        return "rectangle"
    return None

# Tìm biển báo trong video
def findSigns(frame):
    b_img_red, b_img_blue = binaryImg(frame)

    contours_red, _ = cv.findContours(b_img_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv.findContours(b_img_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    used_positions = []

    for c in contours_red + contours_blue:
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)

        shape = is_triangle_or_rectangle(c)
        is_circle = is_circular(c)

        if area > 1500 and (is_circle or shape in ["triangle", "rectangle"]):
            sign = frame[y:y+h, x:x+w]
            label = predict(sign)
            label_text = labelToText.get(label, "Unknown Sign")

            # Tránh chồng chéo vị trí
            text_x, text_y = x, y - 10
            min_distance = 20

            for (px, py) in used_positions:
                if abs(text_y - py) < min_distance:
                    text_y += min_distance

            used_positions.append((text_x, text_y))

            # Vẽ khung và hiển thị tên biển báo
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, label_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

# Xử lý video từ webcam hoặc file video
def process_video(video_source=0):
    cap = cv.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Không thể mở video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Dừng nếu video kết thúc

        processed_frame = findSigns(frame)
        cv.imshow("Traffic Sign Detection", processed_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break  # Nhấn 'q' để thoát

    cap.release()
    cv.destroyAllWindows()

# Chạy nhận diện từ webcam (video_source=0) hoặc từ file ("video.mp4")
process_video("static\videos\video2.mp4")  # Thay 0 bằng "video.mp4" nếu bạn muốn chạy với file video
