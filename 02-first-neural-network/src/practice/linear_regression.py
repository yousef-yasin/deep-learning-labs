import torch
import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print("Loss:", loss.item())
        print("Weight:", model.weight.item())
        print("Bias:", model.bias.item())
        print("--------------------")

test = torch.tensor([[7.0]])
prediction = model(test)

print("\nPrediction for 7:")
print(prediction.item())