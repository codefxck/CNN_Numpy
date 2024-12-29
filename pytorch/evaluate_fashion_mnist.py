import torch
from data_loader import load_data
from models.simple_cnn_fashion_mnist import SimpleCNN_FashionMNIST

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = load_data('fashion_mnist')[1]

    model = SimpleCNN_FashionMNIST()
    model.load_state_dict(torch.load('./models/cnn_model_fashion_mnist.pth', map_location=device, weights_only=True))
    model.to(device)

    accuracy = evaluate_model(model, testloader, device)
    print(f'Fashion-MNIST Model Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()