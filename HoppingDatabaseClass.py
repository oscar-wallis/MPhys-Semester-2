import torch
from DataLoadFunc_v2 import load_data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import os # this is for me as VSCode likes to change the working directory
# print("Working dir:", os.getcwd())
class HoppingsDataset(Dataset):

    def __init__(self, root, npartitions, nkx, nky, nkz, train=None):
        self.root = root
        self.npartitions = npartitions
        self.nkx = nkx
        self.nky = nky
        self.nkz = nkz
        self.train = train

        x, y, z = load_data(root, npartitions, nkx, nky, nkz)

        if self.train:
            x, _, y, _ = train_test_split(x, y, test_size=0.2, random_state=42)
            self.kvals = torch.from_numpy(x.astype(np.float32))
            self.classval = torch.from_numpy(y)
            self.n_samples = x.shape[0]

        if not self.train:
            _, x, _, y = train_test_split(x, y, test_size=0.2, random_state=42)
            self.kvals = torch.from_numpy(x.astype(np.float32))
            self.classval = torch.from_numpy(y)  
            self.n_samples = x.shape[0]


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.kvals[index], self.classval[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
def HoppingsDatabase(batch_size):
    train_dataset = HoppingsDataset('C:/Users/oscar/Documents/Masters ML/Neural Net/NN_data_equal_v1.dat', 1000, 11, 11, 2, train=True) # size [n_samples, n_features]
    test_dataset = HoppingsDataset('C:/Users/oscar/Documents/Masters ML/Neural Net/NN_data_equal_v1.dat', 1000, 11, 11, 2, train=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0)
    return train_loader, test_loader
# train_dataset = HoppingsDataset('NN_data_equal.dat', 1000, 11, 11, 2, train=True) # size [n_samples, n_features]
# train_loader = DataLoader(dataset=train_dataset,batch_size=50, shuffle=True, num_workers=0)
# examples = iter(train_loader)
# example_data, example_targets = next(examples)
# print(example_data.shape, example_targets.shape)
# test_dataset = HoppingsDataset('./Documents/Masters ML/Neural Net/NN_data_sorted.dat', 1000, 11, 11, 2, train=False, transform = transforms.ToTensor())
# test_loader = DataLoader(dataset=test_dataset,batch_size=50, shuffle=True, num_workers=0)