import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Project:
# Student AI Performance Classifier
#
# Inputs:
# study_hours, sleep_hours, attendance, assignments_completed, previous_score
#
# Output:
# 0 = Low
# 1 = Medium
# 2 = High
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

class StudentAIDataset(Dataset):

    def __init__(self):
        self.scale = torch.tensor([10, 8, 100, 5, 100], dtype=torch.float32)

        self.x = torch.tensor([
            [2, 6, 60, 1, 55],
            [3, 5, 65, 2, 60],
            [4, 6, 70, 2, 68],
            [5, 7, 75, 3, 74],
            [6, 7, 80, 3, 78],
            [7, 8, 85, 4, 84],
            [8, 7, 88, 4, 87],
            [9, 8, 92, 5, 93],
            [10, 8, 95, 5, 96],
            [1, 5, 50, 1, 45],
            [2, 5, 55, 1, 50],
            [6, 6, 78, 3, 76],
            [8, 8, 90, 4, 89],
            [9, 7, 94, 5, 94],
            [5, 6, 72, 3, 70],
        ], dtype=torch.float32)

        self.x = self.x / self.scale

        self.y = torch.tensor([
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
            0, 0,
            1,
            2, 2,
            1
        ], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class StudentAIModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.GELU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, dataloader, loss_fn, optimizer, epochs):
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

    return correct / total


def predict_student(model, student_data):
    scale = torch.tensor([10, 8, 100, 5, 100], dtype=torch.float32)

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
    plt.title("Student AI Classifier - Training Loss")
    plt.savefig("../images/student_ai_training_loss.png")
    plt.close()


def main():
    dataset = StudentAIDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True
    )

    model = StudentAIModel()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05
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

    torch.save(model.state_dict(), "../models/student_ai_model.pth")
    print("Model saved successfully.")

    save_loss_plot(loss_history)
    print("Loss plot saved successfully.")

    prediction, probabilities = predict_student(
        model=model,
        student_data=[7, 8, 90, 4, 88]
    )

    print("Probabilities:", probabilities)
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()