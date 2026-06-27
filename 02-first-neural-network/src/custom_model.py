import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 4)

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        x = self.linear2(x)

        return x
    

class MyDataset(Dataset):

    def __init__(self): # to use class one time
        self.x = torch.tensor([[2, 2],
                               [4, 6],
                               [2, 7]], dtype=torch.float32)

        self.y = torch.tensor([[1, 1],
                               [3, 2],
                               [4, 4]], dtype=torch.float32)

    def __len__(self): #to use the len
        return len(self.x)

    def __getitem__(self, index): #to get the index of the linear matrix
        return self.x[index], self.y[index]


dataset = MyDataset()
print(len(dataset))
print(dataset[0])
print(dataset[1])
print(dataset[2])
