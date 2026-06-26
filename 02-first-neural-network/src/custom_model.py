import torch
import torch.nn as nn


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

x = torch.tensor([[2, 2],
                  [4, 6],
                  [2, 7]], dtype=torch.float32)

y = torch.tensor([[1, 1],
                  [3, 2],
                  [4, 4]], dtype=torch.float32)

model = MyModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print("Epoch:", epoch)
        print("Loss:", loss.item())
        print("----------------")

test = torch.tensor([[1.0, 3.0]])
prediction = model(test)

print("Prediction:")
print(prediction)