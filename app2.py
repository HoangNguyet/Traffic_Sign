import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model  # type: ignore
import cv2

# Load the trained CNN model
model = load_model('model2.h5')

# Define HSV ranges for traffic sign colors (điều chỉnh phù hợp với biển báo của bạn)
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Define kernel for morphological operations
kernel_cl = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
], dtype=np.uint8)

# Define classes for traffic signs
classes = {
    1: 'Speed limit (20km/h)',
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
    43: 'End no passing vehicles > 3.5 tons'
}

# Initialize Tkinter
top = tk.Tk()
top.geometry('800x600')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#ffffff')

label = Label(top, background='#ffffff', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to preprocess image for CNN
def preprocess_image(image):
    image = cv2.resize(image, (32, 32))  # Resize to match CNN input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Function to classify traffic signs
def classify_traffic_sign(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_id = np.argmax(predictions)
    return classes[class_id + 1]

# Function to detect traffic signs using HSV thresholding and morphological operations
def detect_traffic_signs(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for red and blue colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine masks
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Apply morphological operations to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_cl)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_cl)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_signs = []
    for contour in contours:
        # Filter contours by area (điều chỉnh ngưỡng phù hợp)
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            detected_signs.append((x, y, w, h))

    return detected_signs

# Function to classify and display results
def classify(file_path):
    global label_packed
    image = cv2.imread(file_path)
    detected_signs = detect_traffic_signs(image)

    result_text = ""
    for (x, y, w, h) in detected_signs:
        # Crop the detected sign
        sign_image_crop = image[y:y+h, x:x+w]

        # Classify the traffic sign
        sign_type = classify_traffic_sign(sign_image_crop)

        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, sign_type, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Append result to text
        result_text += f"Biển báo tại ({x}, {y}): {sign_type}\n"

    # Save the output image
    cv2.imwrite('output_image.jpg', image)

    # Display the result in the GUI
    label.configure(foreground='#011638', text=result_text)

    # Show the output image in the GUI
    output_image = Image.open('output_image.jpg')
    output_image.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(output_image)
    sign_image.configure(image=im)
    sign_image.image = im

# Function to show classify button
def show_classify_button(file_path):
    classify_b = Button(top, text='Nhận dạng', command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

# Function to upload image
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

# GUI setup
upload = Button(top, text='Upload an image', command=upload_image, padx=10, pady=5)
upload.configure(background='#c71b20', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text='Nhận dạng biển báo giao thông', pady=10, font=('arial', 20, 'bold'))
heading.configure(background='#ffffff', foreground='#364156')
heading.pack()

top.mainloop()