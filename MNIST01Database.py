import torch
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
def MNIST01Database(n_samples, batch_size):
    X_train = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    # Leaving only labels 0 and 1 
    idx = np.append(np.where(X_train.targets == 0)[0][:int(0.4*n_samples)], 
                    np.where(X_train.targets == 1)[0][:int(0.4*n_samples)])
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]
    
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

    X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))

    idx = np.append(np.where(X_test.targets == 0)[0][:int(0.1*n_samples)], 
                    np.where(X_test.targets == 1)[0][:int(0.1*n_samples)])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)
    return train_loader, test_loader

