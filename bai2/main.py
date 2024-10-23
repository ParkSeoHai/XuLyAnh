import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang ảnh xám
image = cv2.imread('./image_input.jpg', cv2.IMREAD_GRAYSCALE)

# Phương pháp Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

# Phương pháp Laplacian of Gaussian (LoG)
blurred = cv2.GaussianBlur(image, (3, 3), 0)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(sobel, cmap='gray'), plt.title('Sobel Edge Detection')
plt.subplot(1, 3, 3), plt.imshow(laplacian, cmap='gray'), plt.title('LoG Edge Detection')
plt.show()
