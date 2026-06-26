import torch
import torch.nn as nn

x=torch.tensor([[2,2],[4,6],[2,7]])  #make the tensor array
y=torch.tensor([[1,1],[3,2],[4,4]])

model=nn.Linear(1,1) #make it linear
git=nn.MSELoss() #use loss function in the model(the weight and bias)
optimzer=torch.optim.SGD(model.parameters(),lr=0.01) #to decrease the loss function

for apoche in range(1000) :
    prediction=model(x)
    loss=git(prediction,y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if apoche %10 ==0:
        print(loss.item())
        print(model.weight.item())
        print(model.bias.item())

test=torch.tensor([1,3])
prediction=model(test)
print(prediction)