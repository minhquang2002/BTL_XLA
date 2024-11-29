#The FFT with the numpy.fft module
import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import imageio.v2 as imageio


# Đọc ảnh và chuyển thành ảnh xám
im1 = rgb2gray(imageio.imread('C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\cars.jpg'))

# Tạo figure
plt.figure(figsize=(12, 10))

# Tính biến đổi Fourier của ảnh
freq1 = np.fft.fft2(im1)

# Tính ảnh tái tạo từ biến đổi ngược Fourier
im1_ = np.fft.ifft2(freq1).real

# Hiển thị các kết quả
plt.subplot(2, 2, 1)
plt.imshow(im1, cmap='gray')
plt.title('Original Image', fontsize=20)

# Hiển thị phổ tần số với log scale
plt.subplot(2, 2, 2)
plt.imshow(20 * np.log10(0.01 + np.abs(np.fft.fftshift(freq1))), cmap='gray')
plt.title('FFT Spectrum Magnitude', fontsize=20)

# Hiển thị pha của phổ tần số
plt.subplot(2, 2, 3)
plt.imshow(np.angle(np.fft.fftshift(freq1)), cmap='gray')
plt.title('FFT Phase', fontsize=20)

# Hiển thị ảnh tái tạo sau biến đổi ngược Fourier
plt.subplot(2, 2, 4)
plt.imshow(np.clip(im1_, 0, 255), cmap='gray')
plt.title('Reconstructed Image', fontsize=20)

# Hiển thị figure
plt.show()
