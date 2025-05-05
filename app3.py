import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model  # type: ignore
import cv2

model = load_model('model2.h5')

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing vehicles over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Vehicles > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing vehicles > 3.5 tons'}

top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang grayscale
    image = cv2.resize(image, (32, 32))  # Thay đổi kích thước ảnh đầu vào thành 32x32
    image = np.expand_dims(image, axis=-1)  # Thêm chiều kênh màu (1 kênh)
    image = np.expand_dims(image, axis=0)  # Thêm chiều batch
    image = image / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

    pred_probabilities = model.predict(image)[0]
    pred = pred_probabilities.argmax(axis=-1)
    return classes[pred + 1]

def detect_and_classify_signs(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sử dụng Haar Cascades để phát hiện biển báo
    cascade = cv2.CascadeClassifier('path_to_haarcascade.xml')  # Thay thế bằng đường dẫn đến file Haar Cascade của bạn
    signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in signs:
        sign_image = image[y:y+h, x:x+w]
        sign_type = classify(sign_image)
        results.append((x, y, w, h, sign_type))
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, sign_type, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imwrite('detected_signs.jpg', image)
    return results

def show_classify_button(file_path):
    classify_b = Button(top, text='Nhận dạng', command=lambda: show_results(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def show_results(file_path):
    results = detect_and_classify_signs(file_path)
    result_text = "\n".join([f"Biển báo {i+1}: {sign_type}" for i, (x, y, w, h, sign_type) in enumerate(results)])
    label.configure(foreground='#011638', text=result_text)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(e)

upload = Button(top, text='Upload an image', command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text='Nhận dạng biển báo giao thông', pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')
heading.pack()

top.mainloop()