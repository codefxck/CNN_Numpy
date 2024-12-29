from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(dataset, batch_size=64):
    if dataset == 'fashion_mnist':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.FashionMNIST(root='./data/fashion_mnist/', train=True, download=True,
                                         transform=transform_train)
        testset = datasets.FashionMNIST(root='./data/fashion_mnist/', train=False, download=True,
                                        transform=transform_test)
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='./data/cifar10/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Unsupported dataset")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader