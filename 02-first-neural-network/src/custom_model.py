import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

    def __init__(self):
        self.x = torch.tensor([[2, 2],
                               [4, 6],
                               [2, 7]], dtype=torch.float32)

        self.y = torch.tensor([[1, 1],
                               [3, 2],
                               [4, 4]], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MyModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):

    total_loss = 0

    for batch_x, batch_y in dataloader:

        prediction = model(batch_x)

        loss = loss_fn(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    average_loss = total_loss / len(dataloader)

    if epoch % 50 == 0:
        print("Epoch:", epoch)
        print("Average Loss:", average_loss)
        print("----------------")


test = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
prediction = model(test)

print("Prediction:")
print(prediction)