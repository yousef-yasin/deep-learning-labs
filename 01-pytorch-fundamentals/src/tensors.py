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

# Example 1
a = th.tensor([1,2,3])
print(a+5)  #6,7,8

# Example 2
a = th.tensor([[1,2,3],
               [4,5,6]])
b = th.tensor([10,20,30])
print(a+b)#11,22,33
          #14,25,36

# Example 3
a = th.ones(3,3)
print(a*10)#10 10 10
           #10 10 10
           #10 10 10

# Example 4
a = th.tensor([[1],#(1,3)
               [2],
               [3]])
b = th.tensor([10,20,30])#(1,3)
print(a+b)# 11  21  31
          #12  22   33
          #13 23 33

x = th.tensor(5.0, requires_grad=True)

y = x**2
y.backward()