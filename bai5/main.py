
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# 1. PHÂN LỚP BỘ DỮ LIỆU IRIS
print("Bộ dữ liệu IRIS\n")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb)}")

# CART (Decision Tree with Gini Index)
cart = DecisionTreeClassifier(criterion="gini")
cart.fit(X_train, y_train)
y_pred_cart = cart.predict(X_test)
print(f"CART (Gini) Accuracy: {accuracy_score(y_test, y_pred_cart)}")

# ID3 (Decision Tree with Information Gain)
id3 = DecisionTreeClassifier(criterion="entropy")
id3.fit(X_train, y_train)
y_pred_id3 = id3.predict(X_test)
print(f"ID3 (Information Gain) Accuracy: {accuracy_score(y_test, y_pred_id3)}")

# Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred_mlp)}")

# 2. PHÂN LỚP TRÊN BỘ DỮ LIỆU ẢNH NHA KHOA
print("\nBộ dữ liệu Ảnh Nha Khoa\n")
image_size = (128, 128)  # Resize ảnh cho đồng nhất
data_dir_aug = r"D:\IT3-EAUT\XLA_TGMT\Periapical Lesions\Augmentation JPG Images"
data_dir_orig = r"D:\IT3-EAUT\XLA_TGMT\Periapical Lesions\Original JPG Images"
data_dirs = [data_dir_aug, data_dir_orig]

def load_images_from_directory(directory, label, max_images=100):
    images, labels = [], []
    count = 0
    print(f"Đọc ảnh từ thư mục: {directory}")
    
    if not os.path.exists(directory):
        print(f"Thư mục {directory} không tồn tại.")
        return images, labels

    for filename in os.listdir(directory):
        if count >= max_images:  # Dừng khi đã đọc đủ 100 ảnh
            break
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized.flatten())  # Chuyển ảnh về vector 1 chiều
            labels.append(label)
            count += 1
        else:
            print(f"Không thể đọc ảnh từ {img_path}. Kiểm tra lại định dạng hoặc tên file.")
    
    print(f"Số ảnh đã nạp từ {directory}: {count}")
    return images, labels



# Load dữ liệu ảnh
X_images, y_images = [], []
for i, directory in enumerate(data_dirs):
    images, labels = load_images_from_directory(directory, label=i)
    X_images.extend(images)
    y_images.extend(labels)

X_images = np.array(X_images)
y_images = np.array(y_images)
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_images, y_images, test_size=0.2, random_state=42)

# Huấn luyện các mô hình trên dữ liệu ảnh

# Naive Bayes cho ảnh
nb_img = GaussianNB()
nb_img.fit(X_train_img, y_train_img)
y_pred_nb_img = nb_img.predict(X_test_img)
print(f"Naive Bayes (Image) Accuracy: {accuracy_score(y_test_img, y_pred_nb_img)}")

# CART cho ảnh
cart_img = DecisionTreeClassifier(criterion="gini")
cart_img.fit(X_train_img, y_train_img)
y_pred_cart_img = cart_img.predict(X_test_img)
print(f"CART (Image) Accuracy: {accuracy_score(y_test_img, y_pred_cart_img)}")

# ID3 cho ảnh
id3_img = DecisionTreeClassifier(criterion="entropy")
id3_img.fit(X_train_img, y_train_img)
y_pred_id3_img = id3_img.predict(X_test_img)
print(f"ID3 (Image) Accuracy: {accuracy_score(y_test_img, y_pred_id3_img)}")

# Neural Network cho ảnh
mlp_img = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp_img.fit(X_train_img, y_train_img)
y_pred_mlp_img = mlp_img.predict(X_test_img)
print(f"Neural Network (Image) Accuracy: {accuracy_score(y_test_img, y_pred_mlp_img)}")
