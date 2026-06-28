import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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

        self.scale = torch.tensor([10, 8, 100, 5], dtype=torch.float32)
        self.x = self.x / self.scale

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

loss_history = []

for epoch in range(1500):

    total_loss = 0

    for batch_x, batch_y in dataloader:

        prediction = model(batch_x)

        loss = loss_fn(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    loss_history.append(average_loss)

    if epoch % 100 == 0:
        print("Epoch:", epoch)
        print("Loss:", average_loss)
        print("----------------")


correct = 0
total = 0

model.eval()

with torch.no_grad():
    for x, y in dataloader:
        outputs = model(x)
        predicted = torch.argmax(outputs, dim=1)

        correct += (predicted == y).sum().item()
        total += y.size(0)

accuracy = correct / total

print("Accuracy:", accuracy)


torch.save(model.state_dict(), "student_performance_model.pth")
print("Model saved successfully.")


plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("training_loss.png")
plt.show()


new_student = torch.tensor([[7, 8, 90, 4]], dtype=torch.float32)

new_student = new_student / torch.tensor([10, 8, 100, 5], dtype=torch.float32)

model.eval()

with torch.no_grad():
    output = model(new_student)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print("Probabilities:", probabilities)

if predicted_class.item() == 0:
    print("Prediction: Low")
elif predicted_class.item() == 1:
    print("Prediction: Medium")
else:
    print("Prediction: High")