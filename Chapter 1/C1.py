import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import random_noise

# Đọc ảnh và chuyển thành float
im = imread("C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\fish.jpg")

# Cấu hình figure để vẽ các ảnh
plt.figure(figsize=(15,12))

# Các giá trị sigma khác nhau
sigmas = [0.1, 0.25, 0.5, 1]

# Vẽ ảnh có noise cho mỗi giá trị sigma
for i in range(4):
    noisy = random_noise(im, var=sigmas[i]**2)
    plt.subplot(2, 2, i+1)  # 2 hàng, 2 cột
    plt.imshow(noisy)
    plt.axis('off')  # Tắt trục tọa độ
    plt.title('Gaussian noise with sigma=' + str(sigmas[i]), size=20)

# Điều chỉnh bố cục
plt.tight_layout()
# Hiển thị tất cả các ảnh
plt.show()
