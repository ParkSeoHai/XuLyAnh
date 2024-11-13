import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Định nghĩa các đường dẫn đến thư mục train và validation
train_dir = 'data/train'
validation_dir = 'data/validation'

# Sử dụng ImageDataGenerator để tiền xử lý ảnh và tăng cường dữ liệu (data augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,       # chuẩn hóa pixel từ [0,255] về [0,1]
    rotation_range=40,       # xoay ngẫu nhiên trong khoảng 40 độ
    width_shift_range=0.2,   # dịch ngang ngẫu nhiên
    height_shift_range=0.2,  # dịch dọc ngẫu nhiên
    shear_range=0.2,         # biến đổi shear ngẫu nhiên
    zoom_range=0.2,          # thu phóng ngẫu nhiên
    horizontal_flip=True,    # lật ngang ngẫu nhiên
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Tạo các batch dữ liệu từ thư mục và resize ảnh về 150x150
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'      # phân loại nhị phân (chó, mèo)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # lớp output cho bài toán nhị phân
])

# Compile mô hình
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=8,        # số batch trong mỗi epoch
    epochs=20,                # số epoch (lần lặp huấn luyện toàn bộ dữ liệu)
    validation_data=validation_generator,
    validation_steps=4
)

# Đánh giá mô hình trên tập validation
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {validation_loss}")
print(f"Validation Accuracy: {validation_accuracy}")

# Dự đoán trên ảnh mới
import numpy as np
from tensorflow.keras.preprocessing import image

# Đường dẫn đến ảnh mới
img_path = 'data/test/10049.jpg'

# Tiền xử lý ảnh
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Dự đoán
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("Dự đoán: Chó")
else:
    print("Dự đoán: Mèo")
