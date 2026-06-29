import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Project: Student Score Prediction using PyTorch
# Input: study_hours, sleep_hours
# Output: exam_score
#The relationship between the number of study hours and sleep hours, then predicts a new student's grade.

class StudentDataset(Dataset):#make dataset in your code (Dataset) mean this class is the dataset

    def __init__(self): #it run once
        self.x = torch.tensor([ #the input
            [2, 8],
            [4, 7],
            [6, 7],
            [8, 6],
            [10, 5]
        ], dtype=torch.float32)

        self.y = torch.tensor([ #the output (mean the target)
            [55],
            [65],
            [75],
            [85],
            [95]
        ], dtype=torch.float32)

    def __len__(self): #mean when we need the len(dataset)
        return len(self.x)

    def __getitem__(self, index): #to get the index of the matrix (dataset[0])
        return self.x[index], self.y[index]

#first i make the dataset 

class StudentModel(nn.Module): #its for neural network(will found 1- init 2- forward)

    def __init__(self): #it run once
        super().__init__()

        self.linear1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()      #Activation Function (make all number positive)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, x): #second function
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


dataset = StudentDataset() #first class
dataloader = DataLoader(dataset, batch_size=5, shuffle=True) #to reduce the loss function

model = StudentModel() #second class

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):

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

model.eval() #the training end now only answer

with torch.no_grad(): #dont calc the gradient 
    prediction = model(test) #see the prediction

print("Student study hours: 7")
print("Student sleep hours: 6")
print("Predicted exam score:")
print(prediction.item())