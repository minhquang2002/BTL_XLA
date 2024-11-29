import numpy as np
from scipy import ndimage
from skimage import img_as_float, io
import matplotlib.pyplot as pylab

def rgb2gray(im):
    return np.clip(0.2989 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2], 0, 1)

# Đọc và chuyển ảnh thành dạng grayscale
image_path = 'C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\lena.jpg'
im = rgb2gray(img_as_float(io.imread(image_path)))

# Tạo ảnh làm mờ và ảnh chi tiết
sigma = 5
im_blurred = ndimage.gaussian_filter(im, sigma)
im_detail = np.clip(im - im_blurred, 0, 1)

# Cấu hình hiển thị
pylab.gray()
fig, axes = pylab.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15, 15))
axes = axes.ravel()

# Hiển thị ảnh gốc, ảnh làm mờ và ảnh chi tiết
axes[0].set_title('Original image', size=15)
axes[0].imshow(im)

axes[1].set_title(f'Blurred image, sigma={sigma}', size=15)
axes[1].imshow(im_blurred)

axes[2].set_title('Detail image', size=15)
axes[2].imshow(im_detail)

# Tạo ảnh sắc nét với các giá trị alpha khác nhau
alpha_values = [1, 5, 10]
for i, alpha in enumerate(alpha_values):
    im_sharp = np.clip(im + alpha * im_detail, 0, 1)
    axes[3 + i].imshow(im_sharp)
    axes[3 + i].set_title(f'Sharpened image, alpha={alpha}', size=15)

# Ẩn các trục và sắp xếp giao diện
for ax in axes:
    ax.axis('off')

fig.tight_layout()
pylab.show()