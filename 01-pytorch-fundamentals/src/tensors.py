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