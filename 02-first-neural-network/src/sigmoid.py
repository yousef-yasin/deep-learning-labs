import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class StudentDataset(Dataset):

    def __init__(self):
        self.x = torch.tensor([
            [2, 8],
            [3, 7],
            [4, 6],
            [6, 7],
            [8, 6],
            [10, 5]
        ], dtype=torch.float32)

        # 0 = Fail
        # 1 = Good
        # 2 = Excellent
        self.y = torch.tensor([
            0,
            0,
            1,
            1,
            2,
            2
        ], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StudentModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 8)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        
        return x


dataset = StudentDataset()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = StudentModel()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(500):

    total_loss = 0

    for batch_x, batch_y in dataloader:

        outputs = model(batch_x)

        loss = loss_fn(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    if epoch % 50 == 0:
        print("Epoch:", epoch)
        print("Loss:", average_loss)
        print("-------------------")


test = torch.tensor([[2, 7]], dtype=torch.float32)

model.eval()

with torch.no_grad():
    outputs = model(test)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print("Probabilities:", probabilities)
print("Predicted Class:", predicted_class.item())

if predicted_class.item() == 0:
    print("Fail")
elif predicted_class.item() == 1:
    print("Good")
else:
    print("Excellent")