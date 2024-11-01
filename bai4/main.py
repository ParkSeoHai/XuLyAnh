
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Hàm đọc và tiền xử lý ảnh
def preprocess_image(image_path, image_size=(64, 64)):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Kiểm tra nếu ảnh không đọc được
    if image is None:
        print(f"Không thể đọc được ảnh từ {image_path}")
        return None
    # Thay đổi kích thước ảnh
    image = cv2.resize(image, image_size)
    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    image = image / 255.0
    # Chuyển đổi ảnh thành dạng vector
    return image.flatten()

# Hàm đọc toàn bộ ảnh từ thư mục
def load_images_from_folder(folder_path, image_size=(64, 64)):
    images = []
    labels = []
    # Duyệt qua từng tệp tin trong thư mục
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Tiền xử lý ảnh
        image = preprocess_image(file_path, image_size)
        if image is not None:
            # Thêm ảnh vào danh sách
            images.append(image)
            # Ở đây bạn cần gán nhãn cho ảnh (ví dụ nhãn dựa trên tên tệp hoặc cấu trúc thư mục)
            # Ví dụ: nếu tên tệp bắt đầu bằng "0_", thì gán nhãn 0
            label = 0 if filename.startswith("0_") else 1
            labels.append(label)
    return np.array(images), np.array(labels)

# Đọc ảnh từ thư mục
folder_path = 'D:\\IT3-EAUT\\XLA_TGMT\\train'  # Thay thế bằng đường dẫn tới thư mục của bạn
X, y = load_images_from_folder(folder_path)

# Kiểm tra số lượng ảnh và nhãn
print(f"Số lượng ảnh: {len(X)}, Số lượng nhãn: {len(y)}")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = knn.predict(X_test)
print("Báo cáo phân loại KNN:")
print(classification_report(y_test, y_pred))

# Huấn luyện mô hình SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred_svm = svm.predict(X_test)
print("Báo cáo phân loại SVM:")
print(classification_report(y_test, y_pred_svm))
