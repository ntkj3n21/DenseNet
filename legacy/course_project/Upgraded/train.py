"""
train.py - Hàm huấn luyện, đánh giá và lưu checkpoint

Tham khảo: train.lua và checkpoints.lua trong code gốc.
Chứa các hàm:
  - train_one_epoch(): Huấn luyện 1 epoch
  - evaluate(): Đánh giá trên tập test
  - save_checkpoint(): Lưu trạng thái model ra file .pth
"""

import torch
import os


def train_one_epoch(model, trainloader, optimizer, criterion, device):
    """
    Huấn luyện mô hình trên 1 epoch.

    Args:
        model: Mô hình DenseNet.
        trainloader: DataLoader cho tập huấn luyện.
        optimizer: Bộ tối ưu (Adam, SGD, ...).
        criterion: Hàm mất mát (CrossEntropyLoss).
        device: Thiết bị tính toán (cuda/cpu).

    Returns:
        avg_loss (float): Loss trung bình trên epoch.
        accuracy (float): Độ chính xác (%) trên epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(trainloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, testloader, criterion, device):
    """
    Đánh giá mô hình trên tập kiểm thử.

    Args:
        model: Mô hình DenseNet.
        testloader: DataLoader cho tập kiểm thử.
        criterion: Hàm mất mát.
        device: Thiết bị tính toán (cuda/cpu).

    Returns:
        avg_loss (float): Loss trung bình trên tập test.
        accuracy (float): Độ chính xác (%) trên tập test.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(testloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, dataset_name="cifar10",
                    base_path="/content/drive/MyDrive/DenseNet_Project/results"):
    """
    Lưu trạng thái huấn luyện (checkpoint) ra file .pth.

    Args:
        model: Mô hình đang huấn luyện.
        optimizer: Bộ tối ưu.
        scheduler: Bộ điều chỉnh learning rate.
        epoch (int): Epoch hiện tại.
        dataset_name (str): Tên dataset để phân biệt thư mục lưu.
        base_path (str): Đường dẫn gốc trên Drive.
    """
    path = f"{base_path}/{dataset_name}/checkpoints"
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, f"{path}/model_epoch_{epoch}.pth")
    print(f"💾 Đã lưu Checkpoint [{dataset_name.upper()}] tại epoch {epoch}")
