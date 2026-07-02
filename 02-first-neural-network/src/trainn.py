import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class StudentPerformanceDataset(Dataset):
    def __init__(self):
        self.scale = torch.tensor([10, 8, 100, 5], dtype=torch.float32)

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

        self.x = self.x / self.scale

        self.y = torch.tensor([
            [1],
            [2],
            [2],
            [3],
            [3],
            [4],
            [4],
            [5],
            [5],
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StudentPerformanceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, loss_fn, optimizer, epochs=1500):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        if epoch % 100 == 0:
            print("Epoch:", epoch)
            print("Train Loss:", average_loss)
            print("----------------")


def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            total_loss += loss.item()

    average_test_loss = total_loss / len(test_loader)
    print("Test Loss:", average_test_loss)


def predict_performance(model, student_data):
    scale = torch.tensor([10, 8, 100, 5], dtype=torch.float32)

    student = torch.tensor([student_data], dtype=torch.float32)
    student = student / scale

    model.eval()

    with torch.no_grad():
        output = model(student)

    score = output.item()

    print("Predicted Score:", score)

    if score < 2:
        print("Predicted Performance: Poor")
    elif score < 4:
        print("Predicted Performance: Average")
    else:
        print("Predicted Performance: Excellent")


def main():
    dataset = StudentPerformanceDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False
    )

    model = StudentPerformanceModel()

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    train_model(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=1500
    )

    evaluate_model(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn
    )

    predict_performance(
        model=model,
        student_data=[5, 7, 75, 3]
    )


if __name__ == "__main__":
    main()