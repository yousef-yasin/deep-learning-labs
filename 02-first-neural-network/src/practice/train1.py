import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class StudentDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([[1, 2], [2, 2], [6, 8]], dtype=torch.float32)
        self.y = torch.tensor([[50], [70], [90]], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 7)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(7, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


dataset = StudentDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = StudentModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
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
        print("Average Loss:", average_loss)
        print("----------------")


test = torch.tensor([[7, 6]], dtype=torch.float32)

model.eval()

with torch.no_grad():
    prediction = model(test)

print("Student study hours: 7")
print("Student sleep hours: 6")
print("Predicted exam score:")
print(prediction.item())