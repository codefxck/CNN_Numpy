import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(dataset, batch_size=64):
    if dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.FashionMNIST(root='./data/fashion_mnist/', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data/fashion_mnist/', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='./data/cifar10/', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader