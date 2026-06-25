import torch as th 

x = th.tensor([[1, 2, 3, 4, 5],
              [2,3,4,5,6]])

print("Sum:", x.sum())
print("Shape:", x.shape)
print("Data Type:", x.dtype)

print("Median:", x.median())
print("Min:", x.min())
print("Max:", x.max())
print("Dimensions:", x.ndim)

#to go to any index

print(x[0])
print(x[0][1])
print(x[1,2])
#use slicing
print(x[1,:2])
print(x[0,2:5])

x[0,0]=20
print(x[0])
print("\nZeros:")
print(th.zeros(3,4))

print("\nOnes:")
print(th.ones(2,5))

print("\nRandom:")
print(th.rand(2,3))

print("\nRange:")
print(th.arange(1,11))

print("\nIdentity Matrix:")
print(th.eye(4))
y=th.arange(1,13)
print(y.shape)

z=y.reshape(3,4)
print(z)