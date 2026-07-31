"""
utils.py - Các hàm tiện ích

Chứa hàm vẽ biểu đồ Loss/Accuracy để trực quan hóa kết quả huấn luyện.
"""

import matplotlib.pyplot as plt


def plot_training_history(train_losses, test_losses,
                          train_accuracies, test_accuracies):
    """Vẽ biểu đồ Loss và Accuracy qua các epochs."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.title("Đồ thị Loss qua các Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', marker='s')
    plt.title("Đồ thị Accuracy qua các Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
