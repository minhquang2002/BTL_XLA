from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, rectangle
import matplotlib.pylab as pylab

# Hàm hỗ trợ để hiển thị ảnh với tiêu đề
def plot_image(image, title=''):
    pylab.title(title, size=20)
    pylab.imshow(image)
    pylab.axis('off')  # Ẩn các trục của biểu đồ (có thể bỏ dòng này nếu muốn hiển thị trục)

# Đọc ảnh và chuyển sang grayscale
image_path = 'C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\flowers.jpg'
im = rgb2gray(imread(image_path))

# Chuyển đổi ảnh grayscale sang ảnh nhị phân dựa trên ngưỡng cố định 0.5
im[im <= 0.5] = 0  # Các giá trị <= 0.5 chuyển thành 0 (background)
im[im > 0.5] = 1   # Các giá trị > 0.5 chuyển thành 1 (foreground)

# Hiển thị ảnh gốc và các kết quả xói mòn
pylab.gray()  # Hiển thị ảnh ở chế độ grayscale
pylab.figure(figsize=(20, 10))

# Hiển thị ảnh gốc
pylab.subplot(1, 3, 1)
plot_image(im, 'Original Image')

# Thực hiện phép xói mòn với hình chữ nhật (1, 5)
im_eroded_1 = binary_erosion(im, rectangle(1, 5))
pylab.subplot(1, 3, 2)
plot_image(im_eroded_1, 'Erosion with Rectangle (1, 5)')

# Thực hiện phép xói mòn với hình chữ nhật (1, 15)
im_eroded_2 = binary_erosion(im, rectangle(1, 15))
pylab.subplot(1, 3, 3)
plot_image(im_eroded_2, 'Erosion with Rectangle (1, 15)')

# Hiển thị kết quả
pylab.show()