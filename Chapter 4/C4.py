from skimage.color import rgb2gray
from skimage import exposure
from skimage.io import imread
import pylab

# Đọc ảnh và chuyển đổi sang grayscale
img = rgb2gray(imread('C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\earthfromsky.jpg'))

# Histogram Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Histogram Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Hiển thị ảnh bằng thang màu grayscale
pylab.gray()

# Danh sách ảnh và tiêu đề
images = [img, img_eq, img_adapteq]
titles = [
    'Original Input (Earth from Sky)',
    'After Histogram Equalization',
    'After Adaptive Histogram Equalization'
]

# Hiển thị từng ảnh với tiêu đề
for i in range(3):
    pylab.figure(figsize=(20, 10))
    pylab.imshow(images[i], cmap='gray')
    pylab.title(titles[i], size=15)
    pylab.axis('off')  # Tắt trục tọa độ

# Hiển thị histogram của từng ảnh
pylab.figure(figsize=(15, 5))
for i in range(3):
    pylab.subplot(1, 3, i + 1)
    pylab.hist(images[i].ravel(), color='g', bins=256)
    pylab.title(titles[i], size=15)

pylab.show()