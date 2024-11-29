from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2  # Sử dụng để chuyển đổi màu sắc nếu cần

def detect_objects(image_path):
    """
    Tải mô hình YOLO và chạy dò tìm đối tượng trên ảnh đầu vào.
    """
    model = YOLO("yolov8s.pt")  # Sử dụng mô hình YOLOv8 nhỏ gọn
    results = model(image_path)  # Dò tìm đối tượng
    return results

def display_results(results):
    """
    Hiển thị kết quả dự đoán bằng Matplotlib.
    """
    for result in results:  # Duyệt qua từng kết quả trong danh sách
        annotated_image = result.plot()  # Vẽ bounding boxes lên ảnh

        # Chuyển đổi từ BGR sang RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Hiển thị bằng Matplotlib
        plt.imshow(annotated_image)
        plt.axis("off")  # Tắt trục tọa độ
        plt.show()

def main():
    """
    Chương trình chính: chọn test case và hiển thị kết quả.
    """
    test_cases = {"1": "C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\me4.jpg", "2": "C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\me5.jpg", "3": "C:\\Users\\MinhQuang\\Downloads\\BTL_XLA\\images\\me6.jpg"}
    choice = input("Chọn test case (1, 2, 3): ")
    if choice in test_cases:
        results = detect_objects(test_cases[choice])  # Chạy mô hình và lấy kết quả
        display_results(results)  # Hiển thị kết quả
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()