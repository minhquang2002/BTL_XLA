import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from numpy import sqrt

# Đọc ảnh và loại bỏ kênh alpha (nếu có)
im = imread('C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\swans.jpg')

# Chuyển đổi ảnh RGBA thành RGB bằng cách chỉ lấy 3 kênh đầu tiên
im_rgb = im[..., :3]

# Chuyển ảnh RGB thành ảnh grayscale
im_gray = rgb2gray(im_rgb)

# Phát hiện blob bằng phương pháp Laplacian of Gaussian (LoG)
log_blobs = blob_log(im_gray, max_sigma=30, num_sigma=10, threshold=0.1)
log_blobs[:, 2] = sqrt(2) * log_blobs[:, 2]  # Tính bán kính ở cột thứ 3

# Phát hiện blob bằng phương pháp Difference of Gaussian (DoG)
dog_blobs = blob_dog(im_gray, max_sigma=30, threshold=0.1)
dog_blobs[:, 2] = sqrt(2) * dog_blobs[:, 2]  # Tính bán kính ở cột thứ 3

# Phát hiện blob bằng phương pháp Determinant of Hessian (DoH)
doh_blobs = blob_doh(im_gray, max_sigma=30, threshold=0.005)

# Danh sách các blob và tiêu đề tương ứng
list_blobs = [log_blobs, dog_blobs, doh_blobs]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']

# Vẽ các kết quả
fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharex=True, sharey=True)
axes = axes.ravel()
axes[0].imshow(im, interpolation='nearest')
axes[0].set_title('Original Image', size=30)
axes[0].set_axis_off()

# Duyệt qua các phương pháp và vẽ kết quả
for idx, (blobs, color, title) in enumerate(zip(list_blobs, colors, titles)):
    axes[idx + 1].imshow(im, interpolation='nearest')
    axes[idx + 1].set_title(f'Blobs with {title}', size=30)
    for blob in blobs:
        y, x, radius = blob
        col = plt.Circle((x, y), radius, color=color, linewidth=2, fill=False)
        axes[idx + 1].add_patch(col)
    axes[idx + 1].set_axis_off()

# Hiển thị các kết quả
plt.tight_layout()
plt.show()