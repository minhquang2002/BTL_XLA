import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, find_boundaries
from skimage.io import imread
from skimage.util import img_as_float
from matplotlib.colors import LinearSegmentedColormap

# Danh sách file ảnh
image_files = [
    'C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\victoria_memorial.png',
    'C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\umbc.png',
    'C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\orange.png',
    'C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\mountain.png',
]

for imfile in image_files:
    if not os.path.exists(imfile):
        print(f"File not found: {imfile}")
        continue

    # Đọc và xử lý ảnh
    img = img_as_float(imread(imfile)[::2, ::2, :3])

    # Áp dụng thuật toán Felzenszwalb
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=400)

    # Xác định ranh giới
    borders = find_boundaries(segments_fz)
    unique_colors = np.unique(segments_fz.ravel())
    segments_fz[borders] = -1  # Đánh dấu ranh giới

    # Tính màu trung bình cho từng phân đoạn
    colors = [np.zeros(3)]  # Màu nền
    for color in unique_colors:
        colors.append(np.mean(img[segments_fz == color], axis=0))

    # Tạo colormap
    cm = LinearSegmentedColormap.from_list('palette', colors, N=len(colors))

    # Hiển thị kết quả
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original Image', size=20)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(segments_fz, cmap=cm)
    plt.title('Segmented with Felzenszwalb\'s Method', size=20)
    plt.axis('off')

    plt.show()