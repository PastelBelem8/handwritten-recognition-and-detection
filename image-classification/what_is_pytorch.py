
from __future__ import print_function
import torch

# The creation of an unitialized matrix is just that a declaration!
# Any value that was in the allocated space will be considered the
# initial values of any unitialized matrix, unless explicitly declared. 

# Therefore, executing this line multiple times may yield different results
# as each time the matrix gets allocated in different places in memory
x = torch.empty(5, 3)
print("Empty (undeclared matrix):\n", x)

# This will create a matrix whose values are in the range [0, 1[
x = torch.rand(5, 3)
print("Random matrix\n", x)

# To create a matrix of a specific type, do it like you would in numpy
x = torch.zeros(5, 3, dtype=torch.long)
print("Zero matrix of dtype torch.long:\n", x)

# We can create tensors directly from python lists or numpy arrays as well
x = torch.tensor([5.5, 3])
print("Creating a tensor directly from Python lists:\n", x)

# Or even from another tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

# When creating a tensor from another tensor, Torch infers the type 
# directly. However, in this case, for demonstration purposes we are 
# showing how to override this inference by providing the dtype.
x = torch.randn_like(x, dtype=torch.float) 
print("Created from previous tensor:\n", x)

# To get its size, it is just like numpy
print(x.size(), type(x.size()), "is a tuple")
# Notice that the type of x.size is a tuple and therefore supports all
# the tuple-supported operations

# We can add in multiple ways
y = torch.rand(5,3)

print(x + y)
print(torch.add(x, y))

# torch.add even accepts an output argument
result = torch.empty(5,3)
torch.add(x, y, out=result)
print(result)

# We can also add in place, suffixign the operation with an underscore and applying
# it on the object we want to modify
y.add_(x)  # y will be assigned the result of x + y
print(y)

print("ANY operation that is suffixed with '_' modifies a tensor in place!!")

# To resize a tensor, we use view!
x = torch.randn(4, 4)
print("Before resizing:\n", x) 
y = x.view(16)
print("After flatenning:\n", y)
y = x.view(8,2)
print("After resizing to (8,2):\n", y)
y = x.view(-1, 8)
print("using -1 represents the default. Torch will infer which value to assign to that dimension:\n", y)


# If you have a one element tensor, use x.item() to get its value:
x = torch.tensor([1])
print(x)
print(x.item())

# To learn about more operations on tensors, consult: 
# https://pytorch.org/docs/torch

# To convert numpy to tensor is trivial
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# if we change a, b will also be changed
a.add_(1)
print(a)
print(b)

# To convert from numpy to torch tensor, first create a numpy array 
import numpy as np
a = np.ones(5)
# Then convert it
b = torch.from_numpy(a)

np.add(a, 1, out=a)
print(a)
print(b)

# All tensors except for the CharTensor accept being converted from numpy and back

# To move tensors in and our of GPU we use the objects torch.device
print("Cuda is available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")           # CUDA device object
    y = torch.ones_like(x, device=device)   # directly create a tensor on gpu
    x = y.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
