# 🚀 Hướng Dẫn Setup Môi Trường Nghiên Cứu DenseNet (Đồ Án Machine Learning)

Chào mừng mọi người đến với Repository chính thức của nhóm! 

Repository này chứa toàn bộ mã nguồn và cấu hình môi trường để chúng ta tiến hành huấn luyện, thử nghiệm và đánh giá kiến trúc mạng **DenseNet** trên bộ dữ liệu **CIFAR-10**. Để đảm bảo sự đồng nhất và tránh các lỗi lặt vặt về phiên bản thư viện, toàn bộ nhóm sẽ làm việc trên nền tảng **Google Colab** kết hợp với **Google Drive** cá nhân.

Dưới đây là hướng dẫn chi tiết từng bước để các bạn thiết lập không gian làm việc.

---

## 📁 Cấu trúc Repository hiện tại
* `requirements.txt`: Chứa danh sách các thư viện Python cần thiết (PyTorch, Numpy, Matplotlib,...).
* `DenseNet_Workspace.ipynb`: File Notebook tự động hóa toàn bộ quá trình setup (kết nối Drive, clone code, cài thư viện, tải dataset).

---

## 🛠️ Các bước Setup (Dành cho thành viên)

### Bước 1: Mở File Workspace trên Google Colab
1. Truy cập vào Repository này.
2. Click vào file `DenseNet_Workspace.ipynb`.
3. Nhấn vào nút **"Open in Colab"** (biểu tượng màu xanh) ở đầu trang để mở file này sang giao diện Google Colab của bạn.

### Bước 2: Bật GPU để tăng tốc độ huấn luyện
Huấn luyện mô hình Deep Learning rất nặng, bạn bắt buộc phải bật GPU:
1. Trên thanh menu của Google Colab, chọn **Runtime** (Thời gian chạy) > **Change runtime type** (Thay đổi loại thời gian chạy).
2. Ở mục *Hardware accelerator* (Trình tăng tốc phần cứng), chọn **T4 GPU**.
3. Nhấn **Save** (Lưu).

### Bước 3: Thiết lập bảo mật Git Token (Bắt buộc)
Để Colab có thể kéo (pull) và đẩy (push) code lên Repo này mà không bị lộ mật khẩu:
1. Vào phần **Secrets** (biểu tượng hình chiếc chìa khóa ở thanh công cụ bên trái màn hình Colab).
2. Tạo một secret mới với tên là `GITHUB_TOKEN`.
3. Dán *Personal Access Token* của bạn vào phần giá trị (Value) và bật công tắc cho phép Notebook truy cập.
> ⚠️ **Lưu ý:** Tuyệt đối không copy paste trực tiếp token vào trong code để tránh rò rỉ quyền truy cập tài khoản GitHub!

### Bước 4: Chạy tự động quá trình Setup
Nhấn nút **Run** (biểu tượng nút Play) hoặc nhấn `Shift + Enter` chạy lần lượt các ô code (cells) có sẵn trong file Notebook. Các đoạn code này sẽ tự động làm những việc sau cho bạn:
1. **Kết nối Google Drive:** Hệ thống sẽ yêu cầu quyền truy cập Drive cá nhân của bạn và tạo một thư mục làm việc chung tên là `DenseNet_Project`.
2. **Clone Code:** Tự động tải mới nhất toàn bộ mã nguồn từ thư mục `DoAnML` về máy ảo Colab.
3. **Cài đặt thư viện:** Chạy file `requirements.txt` để đồng bộ thư viện của cả team (bao gồm torch, torchvision, numpy, matplotlib, scikit-learn, tqdm).
4. **Tải Dữ Liệu:** Tự động tải bộ dữ liệu CIFAR-10 (50.000 ảnh train, 10.000 ảnh test) và chuẩn bị sẵn các `DataLoader` (với `batch_size=64`) để sẵn sàng cho việc đưa vào mô hình.

---

## 💻 Bắt đầu Code!

Sau khi ô code cuối cùng báo **"Hoàn tất! Tổng số ảnh huấn luyện: 50000, ảnh kiểm thử: 10000"**, môi trường của bạn đã sẵn sàng 100%!

Công việc tiếp theo của chúng ta:
1. Khởi tạo mô hình DenseNet (sử dụng PyTorch).
2. Viết **Training Loop** (Vòng lặp huấn luyện) sử dụng biến `trainloader` và `testloader` đã được tạo sẵn ở bước 4.
3. Tinh chỉnh các tham số (Learning Rate, Epochs...) và lưu trọng số mô hình (checkpoints) trực tiếp vào Drive cá nhân để không bị mất kết nối giữa chừng.

Chúc cả nhóm code mượt mà, không gặp bug! 🎯
