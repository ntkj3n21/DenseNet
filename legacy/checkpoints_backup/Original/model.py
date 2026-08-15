"""
model.py - Kiến trúc DenseNet GỐC (Original) cho đồ án Machine Learning

Tái tạo lại TRUNG THỰC kiến trúc DenseNet từ bài báo gốc (CVPR 2017)
bằng PyTorch. KHÔNG có bất kỳ nâng cấp nào.

Cấu hình: L=40, k=12 (DenseNet-40 trên CIFAR-10)
Bám sát 100% bài báo gốc:
  - Hàm kích hoạt: ReLU (KHÔNG dùng Mish)
  - Attention: KHÔNG có SE Block
  - Cơ chế nối: torch.cat (concatenation)

Tham khảo:
  - Bài báo gốc: https://arxiv.org/abs/1608.06993
  - Code gốc (Lua/Torch): https://github.com/liuzhuang13/DenseNet
"""

import torch
import torch.nn as nn
import math


# ==============================================================================
# 1. Dense Layer (Lớp đơn trong Dense Block)
# ==============================================================================
class DenseLayer(nn.Module):
    """
    Một lớp đơn bên trong Dense Block.
    Gồm: BatchNorm -> ReLU -> Conv2d(3x3)

    Bám sát bài báo gốc: BN-ReLU-Conv(3x3)
    Tham khảo: densenet.lua dòng 18-26

    Điểm lưu ý: out_channels luôn bằng growth_rate (k=12).
    Dù đầu vào có dày đến đâu, lớp này chỉ xuất ra đúng k bản đồ đặc trưng mới.
    """

    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),  # Bản gốc: dùng ReLU
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

        if self.drop_rate > 0:
            self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = self.layer(x)
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return new_features


# ==============================================================================
# 2. Dense Block (Khối Dày đặc)
# ==============================================================================
class DenseBlock(nn.Module):
    """
    Khối Dense Block - Nơi diễn ra thuật toán chính của DenseNet.

    Chứa một danh sách (ModuleList) các DenseLayer.
    Sau mỗi lớp, đầu ra được NỐI (torch.cat) với đầu vào thay vì CỘNG.
    Đây là điểm khác biệt chí mạng giữa DenseNet và ResNet.

    Tham khảo: densenet.lua dòng 28-47
    """

    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                DenseLayer(layer_in_channels, growth_rate, drop_rate)
            )

    def forward(self, x):
        # Với mỗi lớp, thực hiện phép GHÉP NỐI (concatenation)
        for layer in self.layers:
            new_features = layer(x)
            # KHÔNG ĐƯỢC dùng x = x + new_features (đây là cách của ResNet)
            # PHẢI dùng torch.cat: nối tensor theo chiều channel (dim=1)
            x = torch.cat([x, new_features], dim=1)
        return x


# ==============================================================================
# 3. Transition Layer (Lớp Chuyển tiếp)
# ==============================================================================
class TransitionLayer(nn.Module):
    """
    Sau mỗi Dense Block, tensor đã bị nối thành khối rất dày.
    Lớp Transition có nhiệm vụ "ép cân":
      - BN -> ReLU -> Conv 1x1 để giảm số lượng kênh (channel)
      - AvgPool 2x2 để giảm một nửa chiều dài và chiều rộng

    Tham khảo: densenet.lua dòng 49-58
    """

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),  # Bản gốc: dùng ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


# ==============================================================================
# 4. DenseNet Original (Kiến trúc tổng thể - Bản gốc)
# ==============================================================================
class DenseNetOriginal(nn.Module):
    """
    Kiến trúc DenseNet-40 GỐC cho CIFAR-10.
    Bám sát 100% bài báo: KHÔNG có Mish, KHÔNG có SE Block.

    Tham số:
      growth_rate (int): Tốc độ tăng trưởng k. Mặc định k=12.
      block_config (tuple): Số lớp trong mỗi Dense Block. (12, 12, 12) cho L=40.
      num_classes (int): Số lớp phân loại đầu ra. 10 cho CIFAR-10.
      drop_rate (float): Tỉ lệ dropout. Mặc định 0.0.
      reduction (float): Tỉ lệ nén kênh tại Transition Layer. Mặc định 0.5.

    Cấu trúc tổng thể (theo bài báo gốc):
      Conv đầu vào (3x3) -> [DenseBlock + Transition] x 2
                          -> DenseBlock -> BN + ReLU + AvgPool -> FC

    Tham khảo: densenet.lua dòng 60-90
    """

    def __init__(self, growth_rate=12, block_config=(12, 12, 12),
                 num_classes=10, drop_rate=0.0, reduction=0.5):
        super(DenseNetOriginal, self).__init__()

        # Số kênh ban đầu = 2 * k (theo bài báo gốc - xem densenet.lua dòng 15)
        num_channels = 2 * growth_rate

        # ===== Lớp Convolution đầu vào =====
        # CIFAR-10 dùng Conv 3x3 (khác ImageNet dùng Conv 7x7)
        # Tham khảo: densenet.lua dòng 70
        self.first_conv = nn.Conv2d(3, num_channels, kernel_size=3,
                                    padding=1, bias=False)

        # ===== Xây dựng các Dense Block và Transition Layer =====
        self.features = nn.Sequential()

        for i, num_layers in enumerate(block_config):
            # Thêm Dense Block
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_channels,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock_{i+1}', block)

            # Cập nhật số kênh sau Dense Block
            num_channels = num_channels + num_layers * growth_rate

            # Thêm Transition Layer (trừ block cuối cùng)
            if i != len(block_config) - 1:
                out_channels = int(num_channels * reduction)
                trans = TransitionLayer(num_channels, out_channels)
                self.features.add_module(f'transition_{i+1}', trans)
                num_channels = out_channels

        # ===== Lớp phân loại cuối cùng =====
        # Tham khảo: densenet.lua dòng 42-46, 84
        self.final_bn = nn.BatchNorm2d(num_channels)
        self.final_act = nn.ReLU(inplace=True)  # Bản gốc: dùng ReLU
        self.classifier = nn.Linear(num_channels, num_classes)

        # ===== Khởi tạo trọng số theo cách của bài báo gốc =====
        # Tham khảo: densenet.lua dòng 138-162
        self._initialize_weights()

    def _initialize_weights(self):
        """Khởi tạo trọng số theo phương pháp Kaiming (giống DenseNet gốc)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 1. Convolution đầu vào
        out = self.first_conv(x)

        # 2. Chạy qua các Dense Block + Transition
        out = self.features(out)

        # 3. Batch Normalization + ReLU cuối cùng
        out = self.final_bn(out)
        out = self.final_act(out)

        # 4. Global Average Pooling (ép từ HxW về 1x1)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        # 5. Fully Connected Layer -> Dự đoán lớp
        out = self.classifier(out)
        return out


# ==============================================================================
# Kiểm tra nhanh khi chạy file trực tiếp
# ==============================================================================
if __name__ == '__main__':
    # Tạo model DenseNet-40 GỐC (k=12) cho CIFAR-10
    model = DenseNetOriginal(
        growth_rate=12,
        block_config=(12, 12, 12),
        num_classes=10
    )

    # In tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số của mô hình GỐC: {total_params:,}")

    # Thử chạy một batch giả lập (batch_size=4, 3 kênh màu, ảnh 32x32)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print(f"Kích thước đầu ra: {output.shape}")  # Kỳ vọng: torch.Size([4, 10])
