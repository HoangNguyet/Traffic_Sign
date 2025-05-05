import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from collections import Counter

def augment_data(X_train, y_train, classNo):
    # Đếm số lượng ảnh trong mỗi lớp
    class_counts = Counter(classNo)
    avg_class_count = np.mean(list(class_counts.values()))
    
    # Xác định các lớp có ít ảnh
    underrepresented_classes = {cls: count for cls, count in class_counts.items() if count < avg_class_count}
    
    augmented_images = []
    augmented_labels = []
    
    dataGen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        rotation_range=10
    )
    
    for cls, count in underrepresented_classes.items():
        num_to_generate = int(avg_class_count - count)
        class_indices = np.where(y_train == cls)[0]
        class_images = X_train[class_indices]
        
        gen = dataGen.flow(class_images, np.full(len(class_images), cls), batch_size=1)
        
        for _ in range(num_to_generate):
            new_img, new_label = next(gen)
            augmented_images.append(new_img[0])
            augmented_labels.append(new_label[0])
    
    # Chuyển danh sách sang mảng numpy
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    # Thêm dữ liệu tăng cường vào tập huấn luyện
    X_train = np.concatenate((X_train, augmented_images), axis=0)
    y_train = np.concatenate((y_train, augmented_labels), axis=0)
    
    return X_train, y_train
