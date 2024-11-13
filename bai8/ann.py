import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Đường dẫn đến thư mục dataset
data_dir = "./dataset"  # Thay đổi đường dẫn này theo cấu trúc thư mục của bạn
categories = ["dog", "cat"]  # Các lớp phân loại
image_size = (64, 64)  # Kích thước ảnh chuẩn hóa

data = []
labels = []

# Đọc ảnh từ thư mục 'dog' và 'cat'
for category in categories:
    path = os.path.join(data_dir, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, image_size)  # Resize ảnh về kích thước đồng nhất
            data.append(img_resized)
            labels.append(category)  # Gắn nhãn cho ảnh
        except Exception as e:
            pass

# Chuyển đổi dữ liệu thành numpy array và chuẩn hóa ảnh
data = np.array(data) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

# Mã hóa nhãn (dog: 0, cat: 1)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Chuyển nhãn thành dạng one-hot encoding
labels = to_categorical(labels, num_classes=len(categories))

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Xây dựng mô hình ANN
model = Sequential()

# Thêm lớp Flatten để làm phẳng ảnh đầu vào
model.add(Flatten(input_shape=(image_size[0], image_size[1], 3)))

# Thêm lớp Dense (lớp fully connected)
model.add(Dense(128, activation='relu'))  # Lớp ẩn với 128 nút
model.add(Dense(64, activation='relu'))   # Lớp ẩn với 64 nút

# Lớp output với 2 neuron, mỗi neuron tương ứng với 1 lớp (chó hoặc mèo)
model.add(Dense(len(categories), activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Độ chính xác của ANN: {accuracy * 100:.2f}%")

# Dự đoán một số ảnh
sample_img = cv2.imread(os.path.join(data_dir, 'dog', '10001.jpg'))
sample_img = cv2.resize(sample_img, image_size) / 255.0
sample_img = np.expand_dims(sample_img, axis=0)  # Thêm batch dimension

predicted_class = model.predict(sample_img)
predicted_class = label_encoder.inverse_transform(np.argmax(predicted_class, axis=1))
print(f"Dự đoán lớp ảnh: {predicted_class[0]}")
