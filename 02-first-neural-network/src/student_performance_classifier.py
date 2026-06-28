import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class StudentPerformanceDataset(Dataset):

    def __init__(self):
        self.x = torch.tensor([
            [2, 6, 60, 1],
            [3, 5, 65, 2],
            [4, 6, 70, 2],
            [5, 7, 75, 3],
            [6, 7, 80, 3],
            [7, 8, 85, 4],
            [8, 7, 88, 4],
            [9, 8, 92, 5],
            [10, 8, 95, 5],
        ], dtype=torch.float32)

        self.x = self.x / torch.tensor([10, 8, 100, 5], dtype=torch.float32)

        self.y = torch.tensor([
            0, 0, 0,
            1, 1, 1,
            2, 2, 2
        ], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StudentPerformanceModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4, 16)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        x = self.linear3(x)

        return x


dataset = StudentPerformanceDataset()

dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True
)

model = StudentPerformanceModel()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(500):

    total_loss = 0

    for batch_x, batch_y in dataloader:

        prediction = model(batch_x)

        loss = loss_fn(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    if epoch % 50 == 0:
        print("Epoch:", epoch)
        print("Loss:", average_loss)
        print("----------------")