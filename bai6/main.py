import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Tải bộ dữ liệu IRIS
iris = load_iris()
X = iris.data  # Các đặc trưng
y = iris.target  # Nhãn thực tế

def kmeans(X, k, max_iters=100):    # max_iters: Số lần lặp tối đa cho thuật toán.
    # Khởi tạo trung tâm ngẫu nhiên cho k cụm
    np.random.seed(42)  # Để tái tạo kết quả
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Gán nhãn cho mỗi điểm dữ liệu
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Cập nhật trung tâm cụm
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Kiểm tra xem trung tâm có thay đổi hay không
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

k = 3  # Số cụm
predicted_labels, centroids = kmeans(X, k)    # mảng chứa tọa độ của các trung tâm cụm.
# predicted_labels:Mảng chứa nhãn dự đoán cho từng mẫu trong bộ dữ liệu dựa trên khoảng cách từ mẫu đến các trung tâm cụm.

def calculate_f1_score(true_labels, predicted_labels):
    # Tính toán số lượng nhãn đúng
    tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    fp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != pred and pred in true_labels)
    fn = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != pred and true in true_labels)

    # Tính Precision và Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Tính F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

# Ánh xạ các nhãn
clusters = {i: [] for i in range(k)}
for index, label in enumerate(predicted_labels):
    clusters[label].append(y[index])

mapped_labels = []
for i in range(k):
    most_common_label = np.bincount(clusters[i]).argmax()  # Nhãn phổ biến nhất
    mapped_labels.append(most_common_label)

final_predicted_labels = [mapped_labels[label] for label in predicted_labels]

# Tính toán F1-score
f1 = calculate_f1_score(y, final_predicted_labels)
print(f"F1 Score: {f1}")

# Trực quan hóa các cụm
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering on IRIS Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

