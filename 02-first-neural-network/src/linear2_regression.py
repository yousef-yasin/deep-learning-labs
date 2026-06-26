import torch
import torch.nn as nn

x=torch.tensor([[2,2],[4,6],[2,7]], dtype=torch.float32)  #make the tensor array
y=torch.tensor([[1,1],[3,2],[4,4]], dtype=torch.float32)

model=nn.Linear(2,2) #make it linear
git=nn.MSELoss() #use loss function in the model(the weight and bias)
optimzer=torch.optim.SGD(model.parameters(),lr=0.01) #to decrease the loss function

for apoche in range(100) :
    prediction=model(x)
    loss=git(prediction,y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if apoche %10 ==0:
        print(loss.item())
        print(model.weight)
        print(model.bias)

test=torch.tensor([[1.0,3.0]])
prediction=model(test)
print(prediction)