import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Chuyển ảnh thành dạng vector (1 chiều) để sử dụng trong SVM
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Khởi tạo mô hình SVM với kernel là RBF (Radial Basis Function)
svm = SVC(kernel='rbf', gamma='scale')

# Huấn luyện mô hình với dữ liệu huấn luyện
svm.fit(x_train_flat, y_train)

# Dự đoán trên tập kiểm tra
y_pred_svm = svm.predict(x_test_flat)

# Đánh giá độ chính xác
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Độ chính xác của SVM: {accuracy_svm * 100:.2f}%")
