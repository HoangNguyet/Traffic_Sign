import cv2
import numpy as np
from tkinter import Tk, filedialog

def nothing(x):
    pass

# Tạo hộp thoại chọn file
Tk().withdraw()  # Ẩn cửa sổ chính của tkinter
file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])

if not file_path:
    print("❌ Không có ảnh nào được chọn.")
    exit()

# Đọc ảnh
img = cv2.imread(file_path)
if img is None:
    print("❌ Lỗi khi đọc ảnh.")
    exit()

cv2.namedWindow('image')

# Tạo các trackbar để chỉnh HSV
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set mặc định cho Max HSV
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Biến lưu giá trị trước đó
phMin = psMin = pvMin = phMax = psMax = pvMax = 0
waitTime = 33

while True:
    # Lấy giá trị hiện tại của các trackbar
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Tạo mask theo ngưỡng HSV
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    # In ra nếu có thay đổi
    if (phMin != hMin or psMin != sMin or pvMin != vMin or
        phMax != hMax or psMax != sMax or pvMax != vMax):
        print(f"(hMin={hMin}, sMin={sMin}, vMin={vMin}) → (hMax={hMax}, sMax={sMax}, vMax={vMax})")
        phMin, psMin, pvMin = hMin, sMin, vMin
        phMax, psMax, pvMax = hMax, sMax, vMax

    # Hiển thị ảnh kết quả
    cv2.imshow('image', output)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
