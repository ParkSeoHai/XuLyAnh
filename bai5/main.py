import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Đọc bộ dữ liệu IRIS
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Đọc dữ liệu ảnh từ các tệp
def load_images_from_folder(folder, limit=100):
    images = []
    filenames = []
    count = 0
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') and count < limit:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
                count += 1
    return images, filenames

folder1 = r"D:\IT3-EAUT\XLA_TGMT\train\AugmentationJPGImages"
folder3 = r"D:\IT3-EAUT\XLA_TGMT\train\OriginalJPGImages"
images1, filenames1 = load_images_from_folder(folder1)
images3, filenames3 = load_images_from_folder(folder3)

# Đọc dữ liệu annotation từ file XML
def load_annotations_from_xml(folder, filenames):
    annotations = []
    for filename in filenames:
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        xml_path = os.path.join(folder, xml_filename)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            label = root.find('label').text if root.find('label') else None
            if label:
                annotations.append(label)
            else:
                annotations.append('Unknown')  # Gán nhãn mặc định nếu không có nhãn
        else:
            annotations.append('Unknown')  # Gán nhãn mặc định nếu không tìm thấy tệp XML
    return annotations

folder2 = r"D:\IT3-EAUT\XLA_TGMT\train\ImageAnnots"
annotations1 = load_annotations_from_xml(folder2, filenames1)
annotations3 = load_annotations_from_xml(folder2, filenames3)

# Chuyển đổi ảnh và annotation thành các đặc trưng đơn giản
def preprocess_images(images):
    features = []
    for img in images:
        img_resized = cv2.resize(img, (64, 64))  # Giảm kích thước ảnh
        img_flattened = img_resized.flatten()  # Chuyển ảnh thành vector
        features.append(img_flattened)
    return np.array(features)

X_images1 = preprocess_images(images1)
X_images3 = preprocess_images(images3)

# Kiểm tra tính nhất quán của dữ liệu
assert len(X_images1) == len(annotations1), "Số lượng ảnh và nhãn không khớp nhau trong folder1!"
assert len(X_images3) == len(annotations3), "Số lượng ảnh và nhãn không khớp nhau trong folder3!"

# Ghép dữ liệu ảnh và nhãn thành một bộ dữ liệu
X_images = np.concatenate((X_images1, X_images3), axis=0)
y_images = annotations1 + annotations3  # Ghép nhãn cho cả hai thư mục

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_images, y_images, test_size=0.2, random_state=42)

# Naive Bayes (cho bộ dữ liệu IRIS)
nb = GaussianNB()
nb.fit(X_train_iris, y_train_iris)
y_pred_iris_nb = nb.predict(X_test_iris)
print("Naive Bayes IRIS Accuracy:", accuracy_score(y_test_iris, y_pred_iris_nb))

# CART (cho bộ dữ liệu ảnh, đo lường bằng Gini Index)
cart = DecisionTreeClassifier(criterion="gini")
cart.fit(X_train_img, y_train_img)
y_pred_img_cart = cart.predict(X_test_img)
print("CART Image Accuracy:", accuracy_score(y_test_img, y_pred_img_cart))

# ID3 (cho bộ dữ liệu IRIS, đo lường bằng Information Gain)
id3 = DecisionTreeClassifier(criterion="entropy")
id3.fit(X_train_iris, y_train_iris)
y_pred_iris_id3 = id3.predict(X_test_iris)
print("ID3 IRIS Accuracy:", accuracy_score(y_test_iris, y_pred_iris_id3))

# Neural Network (cho bộ dữ liệu ảnh)
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
nn.fit(X_train_img, y_train_img)
y_pred_img_nn = nn.predict(X_test_img)
print("Neural Network Image Accuracy:", accuracy_score(y_test_img, y_pred_img_nn))
