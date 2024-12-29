import numpy as np
from data_loader import load_data
from models.simple_cnn_cifar10 import SimpleCNN_CIFAR10

# 加载数据
_, testloader_cifar = load_data('cifar10', batch_size=64)

# 加载已训练的模型
model = SimpleCNN_CIFAR10()
model.load_state_dict(np.load('cnn_model_cifar10.npy', allow_pickle=True).item())

# 评估模型
def evaluate(model, testloader):
    correct = 0
    total = 0
    for inputs, labels in testloader:
        outputs = model.forward(inputs.numpy())
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()

    print(f'CIFAR-10 Model Accuracy: {100 * correct / total:.2f}%')

# 评估模型
evaluate(model, testloader_cifar)