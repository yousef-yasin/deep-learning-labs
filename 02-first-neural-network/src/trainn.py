import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

x=torch.tensor([[2, 6, 60, 1],
                 [3, 5, 65, 2],
                    [4, 6, 70, 2],
                    [5, 7, 75, 3],
                    [6, 7, 80, 3],
                    [7, 8, 85, 4],
                    [8, 7, 88, 4],
                    [9, 8, 92, 5],
                    [10, 8, 95, 5]], dtype=torch.float32)

y=torch.tensor([[1], [2], [2], [3], [3], [4], [4], [5], [5]], dtype=torch.float32)

class StudentPerformanceDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1))

for epoch in range(1500):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    dataset = StudentPerformanceDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

trained_model = model
test_input = torch.tensor([[5, 7, 75, 3]], dtype=torch.float32)
with torch.no_grad():
    test_output = trained_model(test_input)
    print(f'Test Input: {test_input}, Predicted Output: {test_output.item()}')