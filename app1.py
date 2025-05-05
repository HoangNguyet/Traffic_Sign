from __future__ import division, print_function
import sys
import os
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model # type: ignore
from flask import Flask, request, render_template, Response, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Tải mô hình nhận diện biển báo giao thông
model = load_model('model_demo.h5', compile=False)

# Hàm tiền xử lý hình ảnh
def preprocessing(img):
    img = img / 255.0
    return img

# Hàm trả về tên của biển báo giao thông
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left',
        'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo]

# Hàm xử lý và dự đoán biển báo trên từng khung hình
def predict_frame(frame, model):
    img = cv2.resize(frame, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 3)
    
    predictions, bbox_pred = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    label = getClassName(classIndex)
    bbox = bbox_pred[0]  # [x1, y1, x2, y2]

    # Vẽ bounding box và nhãn
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return frame

# Hàm xử lý video và truyền khung hình về trình duyệt
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = predict_frame(frame, model)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

# Giao diện chính
@app.route('/', methods=['GET'])
def index():
    filename = request.args.get('filename', 'video1.mp4')
    if not filename:
        filename = 'video1.mp4'
    return render_template('index1.html', filename=filename)

# Xử lý upload video và hiển thị kết quả
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    f = request.files['file']
    if f.filename == '':
        return 'No selected file'
    
    if f:
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        
        return redirect(url_for('index', filename=secure_filename(f.filename)))

    return 'Invalid request'

# Truyền dữ liệu video ra trình duyệt
@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('uploads', filename)
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
