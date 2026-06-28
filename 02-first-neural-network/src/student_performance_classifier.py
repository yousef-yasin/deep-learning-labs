import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, dataloader, loss_fn, optimizer, epochs=1500):
    loss_history = []

    for epoch in range(epochs):
        model.train()
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

    return loss_history


def evaluate_model(model, dataloader):
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
    return accuracy


def predict_student(model, student_data):
    scale = torch.tensor([10, 8, 100, 5], dtype=torch.float32)

    student = torch.tensor([student_data], dtype=torch.float32)
    student = student / scale

    model.eval()

    with torch.no_grad():
        output = model(student)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    class_names = ["Low", "Medium", "High"]

    return class_names[predicted_class.item()], probabilities


def save_loss_plot(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("training_loss.png")
    plt.close()


def main():
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

    loss_history = train_model(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=1500
    )

    accuracy = evaluate_model(model, dataloader)

    print("Final Accuracy:", accuracy)

    torch.save(model.state_dict(), "student_performance_model.pth")
    print("Model saved as student_performance_model.pth")

    save_loss_plot(loss_history)
    print("Loss plot saved as training_loss.png")

    prediction, probabilities = predict_student(
        model=model,
        student_data=[7, 8, 90, 4]
    )

    print("Probabilities:", probabilities)
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()