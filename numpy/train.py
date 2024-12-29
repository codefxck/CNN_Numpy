import os
import sys
import numpy as np
import time
from data_loader import load_data
from utils.plot import plot_loss_curve
from models.simple_cnn_fashion_mnist import SimpleCNN_FashionMNIST
from models.simple_cnn_cifar10 import SimpleCNN_CIFAR10


# 定义交叉熵损失函数
def cross_entropy_loss(outputs, labels):
    m = labels.shape[0]
    p = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    p /= np.sum(p, axis=1, keepdims=True)
    p = np.clip(p, 1e-10, 1.0)  # 确保概率不为零，避免 log(0)
    log_likelihood = -np.log(p[range(m), labels])
    loss = np.sum(log_likelihood) / m
    grad = p
    grad[range(m), labels] -= 1
    grad /= m
    return loss, grad


# 训练模型函数
def train_model(model, trainloader, num_epochs=20, lr=0.01, log_interval=10):
    loss_history = []
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            # 前向传播
            outputs = model.forward(inputs.numpy())
            loss, grad_output = cross_entropy_loss(outputs, labels.numpy())

            # 反向传播和参数更新
            model.backward(grad_output, lr)

            running_loss += loss

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(trainloader)}], Loss: {running_loss / log_interval:.3f}')
                running_loss = 0.0

        # 记录每个epoch的平均损失
        epoch_loss = running_loss / len(trainloader)
        loss_history.append(epoch_loss)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training time for {num_epochs} epochs: {elapsed_time:.2f} seconds')

    return loss_history


def main():
    dataset = input("请选择要训练的数据集（fashion_mnist 或 cifar10）：")
    if dataset == 'fashion_mnist':
        model = SimpleCNN_FashionMNIST()
        model_save_path = './models/cnn_model_fashion_mnist.npy'
    elif dataset == 'cifar10':
        model = SimpleCNN_CIFAR10()
        model_save_path = './models/cnn_model_cifar10.npy'
    else:
        print("无效输入，请输入 'fashion_mnist' 或 'cifar10'")
        return

    trainloader, testloader = load_data(dataset, batch_size=64)

    print(f"Training on {dataset} dataset...")
    loss_history = train_model(model, trainloader, num_epochs=3)

    # 保存模型参数
    model_params = {
        'conv1_kernels': model.conv1.kernels,
        'conv1_biases': model.conv1.biases,
        'conv2_kernels': model.conv2.kernels,
        'conv2_biases': model.conv2.biases,
        'conv3_kernels': model.conv3.kernels,
        'conv3_biases': model.conv3.biases,
        'fc1_weights': model.fc1.weights,
        'fc1_biases': model.fc1.biases,
        'fc2_weights': model.fc2.weights,
        'fc2_biases': model.fc2.biases
    }
    np.save(model_save_path, model_params)
    print(f"Model saved to {model_save_path}")

    plot_loss_curve(loss_history, save_path=f'./plots/training_loss_curve_{dataset}.png')


if __name__ == "__main__":
    main()