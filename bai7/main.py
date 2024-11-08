import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào và chuyển sang ảnh xám
image = cv2.imread('./image-map.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng Gaussian Blur để giảm nhiễu
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# 1. Toán tử Sobel
sobelx = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)  # Cạnh dọc
sobely = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)  # Cạnh ngang
sobel_combined = cv2.magnitude(sobelx, sobely)  # Tổng hợp

# 2. Toán tử Prewitt
# Tạo kernel Prewitt và áp dụng
prewittx = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))  # Dọc
prewitty = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))  # Ngang

# Tính toán tổng hợp gradient
prewitt_combined = cv2.magnitude(prewittx, prewitty)

# 3. Toán tử Roberts
# Áp dụng bộ lọc Roberts với kiểu dữ liệu float32
robertsx = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[1, 0], [0, -1]]))  # Chéo 1
robertsy = cv2.filter2D(gaussian_blur, cv2.CV_32F, np.array([[0, 1], [-1, 0]]))  # Chéo 2

# Tính toán tổng hợp gradient
roberts_combined = cv2.magnitude(robertsx, robertsy)

# 4. Toán tử Canny
canny_edges = cv2.Canny(gaussian_blur, 100, 200)

# Hiển thị các kết quả
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray')
plt.title("Ảnh gốc"), plt.axis("off")

plt.subplot(2, 3, 2), plt.imshow(sobel_combined, cmap='gray')
plt.title("Toán tử Sobel"), plt.axis("off")

plt.subplot(2, 3, 3), plt.imshow(prewitt_combined, cmap='gray')
plt.title("Toán tử Prewitt"), plt.axis("off")

plt.subplot(2, 3, 4), plt.imshow(roberts_combined, cmap='gray')
plt.title("Toán tử Roberts"), plt.axis("off")

plt.subplot(2, 3, 5), plt.imshow(canny_edges, cmap='gray')
plt.title("Toán tử Canny"), plt.axis("off")

plt.subplot(2, 3, 6), plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blur"), plt.axis("off")

plt.tight_layout()
plt.show()
