import torch
import torch.optim as optim
from data_loader import load_data
from utils.plot import plot_loss_curve
from models.simple_cnn_fashion_mnist import SimpleCNN_FashionMNIST
from models.simple_cnn_cifar10 import SimpleCNN_CIFAR10

def train_model(model, trainloader, criterion, optimizer, num_epochs=5, log_interval=10):
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % log_interval == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / log_interval:.3f}')
                running_loss = 0.0

        epoch_loss = running_loss / len(trainloader)
        loss_history.append(epoch_loss)

    return loss_history

def main():
    dataset = input("请选择要训练的数据集（fashion_mnist 或 cifar10）：")
    if dataset == 'fashion_mnist':
        model = SimpleCNN_FashionMNIST()
        model_save_path = './models/cnn_model_fashion_mnist.pth'
    elif dataset == 'cifar10':
        model = SimpleCNN_CIFAR10()
        model_save_path = './models/cnn_model_cifar10.pth'
    else:
        print("无效输入，请输入 'fashion_mnist' 或 'cifar10'")
        return

    trainloader, testloader = load_data(dataset)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {dataset} dataset...")
    loss_history = train_model(model, trainloader, criterion, optimizer)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_loss_curve(loss_history, save_path=f'./plots/training_loss_curve_{dataset}.png')

if __name__ == "__main__":
    main()