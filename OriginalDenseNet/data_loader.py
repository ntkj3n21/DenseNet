"""
data_loader.py - Tải và xử lý dữ liệu cho DenseNet GỐC

Hỗ trợ 3 bộ dữ liệu:
  - CIFAR-10:  10 lớp, ảnh 32x32 (mặc định)
  - CIFAR-100: 100 lớp, ảnh 32x32
  - SVHN:      10 lớp, ảnh 32x32 (Street View House Numbers)

Tham khảo: dataloader.lua và datasets/cifar10.lua trong code gốc.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


# Mean/Std chuẩn hóa riêng cho từng dataset (tính từ tập train)
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
}

# Số lớp phân loại cho từng dataset
DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
}


def get_data_loaders(dataset_name="cifar10", data_dir="./data",
                     batch_size=64, num_workers=2):
    """
    Tải bộ dữ liệu và trả về DataLoader cho train và test.

    Args:
        dataset_name (str): Tên dataset ("cifar10", "cifar100", "svhn").
        data_dir (str): Thư mục gốc chứa dữ liệu.
        batch_size (int): Kích thước batch. Mặc định 64.
        num_workers (int): Số luồng tải dữ liệu song song.

    Returns:
        trainloader, testloader: DataLoader cho tập huấn luyện và kiểm thử.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Dataset '{dataset_name}' không được hỗ trợ. "
                         f"Chọn: {list(DATASET_STATS.keys())}")

    stats = DATASET_STATS[dataset_name]
    mean, std = stats["mean"], stats["std"]

    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    if dataset_name == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=dataset_dir, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(
            root=dataset_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=dataset_dir, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(
            root=dataset_dir, train=False, download=True, transform=test_transform)
    elif dataset_name == "svhn":
        trainset = torchvision.datasets.SVHN(
            root=dataset_dir, split="train", download=True, transform=train_transform)
        testset = torchvision.datasets.SVHN(
            root=dataset_dir, split="test", download=True, transform=test_transform)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    num_classes = DATASET_NUM_CLASSES[dataset_name]
    print(f"📦 Dataset: {dataset_name.upper()} | "
          f"Train: {len(trainset)} ảnh | Test: {len(testset)} ảnh | "
          f"Số lớp: {num_classes}")

    return trainloader, testloader
