import numpy as np
from data_loader import load_data
from models.simple_cnn_fashion_mnist import SimpleCNN_FashionMNIST

# 加载数据
_, testloader_fashion = load_data('fashion_mnist', batch_size=64)

# 加载已训练的模型
model = SimpleCNN_FashionMNIST()
model.load_state_dict(np.load('cnn_model_fashion_mnist.npy', allow_pickle=True).item())

# 评估模型
def evaluate(model, testloader):
    correct = 0
    total = 0
    for inputs, labels in testloader:
        outputs = model.forward(inputs.numpy())
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()

    print(f'Fashion-MNIST Model Accuracy: {100 * correct / total:.2f}%')

# 评估模型
evaluate(model, testloader_fashion)