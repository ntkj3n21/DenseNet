# OriginalDenseNetPythonReplica

Tái tạo trung thực kiến trúc **DenseNet GỐC** (CVPR 2017, Best Paper Award) bằng Python/PyTorch.

## Đặc điểm
- **Bám sát 100% bài báo gốc**: ReLU, KHÔNG có Mish, KHÔNG có SE Block
- **Cấu hình**: DenseNet-40 (L=40, k=12)
- **Dataset**: CIFAR-10, CIFAR-100, SVHN

## Cấu trúc dự án
| File | Chức năng |
|------|----------|
| `model.py` | Kiến trúc DenseNet GỐC (DenseLayer, DenseBlock, TransitionLayer, DenseNetOriginal) |
| `data_loader.py` | Tải và xử lý dữ liệu CIFAR-10, CIFAR-100, SVHN |
| `train.py` | Hàm huấn luyện, đánh giá, lưu checkpoint |
| `utils.py` | Vẽ biểu đồ Loss/Accuracy |
| `DenseNet_Workspace.ipynb` | Notebook huấn luyện trên Google Colab |

## So sánh với bản Nâng cấp (DoAnML)
| Thành phần | Bản gốc (repo này) | Bản nâng cấp (DoAnML) |
|---|---|---|
| Hàm kích hoạt | ReLU | Mish |
| Attention | Không có | SE Block |
| Class name | `DenseNetOriginal` | `DenseNetCustom` |
