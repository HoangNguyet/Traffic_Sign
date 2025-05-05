import numpy as np
import cv2 as cv
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model  # type: ignore
import cv2

cap = cv.VideoCapture("static/videos/video39.mp4") 
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

model = load_model("model_26.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

kernel_ci = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]], dtype=np.uint8)

# HSV image
def returnHSV(img):
    blur = cv.GaussianBlur(img, (5,5), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    return hsv

# Binary the img from HSV range
def binaryImg(img):
    image1 = img.copy()
    image2 = img.copy()
    image_blue = img.copy()
    
    hsv1 = returnHSV(image1)
    hsv2 = returnHSV(image2)
    hsvblue = returnHSV(image_blue)
    
    b_img1 = cv.inRange(hsv1, low_thresh1, high_thresh1)
    b_img2 = cv.inRange(hsv2, low_thresh2, high_thresh2)
    
    # Binary resign img
    b_img_red = cv.bitwise_or(b_img1, b_img2)
    
    # Binary blue sign img
    b_img_blue = cv.inRange(hsvblue, low_thresh3, high_thresh3)
    
    return b_img_red, b_img_blue

def findContour(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def boundaryBox(img, contours):
    box = cv.boundingRect(contours)
    sign = img[box[1]:(box[1] + box[3]), box[0]: (box[0] + box[2])]
    return img, sign, box

# Preprocessing img
def preprocessingImageToClassifier(image=None, imageSize=32, mu=102.23982103497072, std=72.11947698025735):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (imageSize, imageSize))
    image = (image - mu) / std
    image = image.reshape(1, imageSize, imageSize, 1)
    return image

def predict(sign):
    img = preprocessingImageToClassifier(sign, imageSize=32)
    return np.argmax(model.predict(img))

# Finding the red sign
def findRedSign(frame):
    b_img_red, _ = binaryImg(frame)
    contours = findContour(b_img_red)
    for c in contours:
        area = cv.contourArea(c)
        if area > 500:
            (a, b), r = cv.minEnclosingCircle(c)
            
            # Checking the round shape or triangle shape of red sign
            if area > 0.42 * np.pi * r * r:
                img, sign, box = boundaryBox(frame, c)
                x, y, w, h = box
                
                # Checking the distance of top and bottom, aspect ratio of triangle and round shape
                if (w / h > 0.7) and (w / h < 1.2) and ((y + h) < 0.6 * height) and (y > height / 20):
                    label = labelToText[predict(sign)]
                    box = np.asarray(box)
                    rois.append(box)
                    labels.append(label)

# Finding the blue sign
def findingBlueSign(frame):
    _, b_img_blue = binaryImg(frame)
    contours_blue = findContour(b_img_blue)
    for c_blue in contours_blue:
        area_blue = cv.contourArea(c_blue)
        if area_blue > 1200:
            (a, b), r = cv.minEnclosingCircle(c_blue)
            area_circle = np.pi * r * r
            
            # Checking the round shape of blue sign
            if area_blue > 0.7 * area_circle:
                _, sign, box = boundaryBox(frame, c_blue)
                x, y, w, h = box
                
                # Checking the distance of top and bottom: aspect ratio
                if (w / h > 0.77) and (w / h < 1.2) and (y + h) < 0.6 * height:
                    label = labelToText[predict(sign)]
                    box = np.asarray(box)
                    rois.append(box)
                    labels.append(label)

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

# Red
low_thresh1 = (165, 100, 40)
high_thresh1 = (179, 255, 255)

low_thresh2 = (0, 160, 40)
high_thresh2 = (10, 255, 255)

# Blue
low_thresh3 = (100, 150, 40)
high_thresh3 = (130, 255, 255)

isTracking = 0
frame_count = 0
max_trackingFrame = 10

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Can't read video frame")
        break
    
    height, width, _ = frame.shape
    
    if isTracking == 0:
        # Run detection code
        rois = []
        labels = []
        findRedSign(frame)
        findingBlueSign(frame)
        
        # Re-create and initialize the tracker
        trackers = cv.legacy.MultiTracker_create()
        for roi in rois:
            trackers.add(cv.legacy.TrackerCSRT_create(), frame, tuple(roi))
        isTracking = 1
    else:
        if frame_count == max_trackingFrame:
            isTracking = 0
            frame_count = 0
        # Update object location
        ret, objs = trackers.update(frame)
        if ret:
            label_count = 0
            for obj in objs:
                # Draw bounding box and label
                p1 = (int(obj[0]), int(obj[1]))
                p2 = (int(obj[0] + obj[2]), int(obj[1] + obj[3]))
                cv.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv.rectangle(frame, p1, (int(obj[0] + 2 * obj[2]), int(obj[1] - 15)), (0, 255, 0), -1)
                cv.putText(frame, labels[label_count], (int(obj[0] + (obj[2] / 2) - 5), int(obj[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                label_count += 1
        else:
            print("Tracking fail")
            isTracking = 0
        frame_count += 1
    
    cv.imshow('video', frame)
    if cv.waitKey(10) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()