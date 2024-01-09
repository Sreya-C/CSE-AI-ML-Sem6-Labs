import torch

# print(torch.__version__)

tensor = torch.tensor([[[1,1,1],[2,2,2],[3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]])
print(tensor.ndim)
print(tensor.shape)
print(tensor)

#Reshaping a Tensor
print("Reshaping")
print(tensor.reshape(6,3))
print()

#Viewing - Returns a new tensor with the same data as the self tensor but of a different shape.
print("Viewing")
x = tensor.view(18)
print(x)
print()

#Stacking
print("Stacking")
x = torch.stack((torch.tensor([1,2,3]),torch.tensor([4,5,6])))
print(x)
print()

#Squeeze - Returns a tensor with all specified dimensions of input of size 1 is removed.
print("Squeeze")
x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())
y = torch.squeeze(x)
print(y.size())
y = torch.squeeze(x, 3) #Checks for size 1 only at index 3
print(y.size())
print()

#Unsqueezing - Returns a new tensor with a dimension of size one inserted at the specified position.
print("Unsqueeze")
x = torch.zeros(2,2)
print(x)
print(torch.unsqueeze(x, 0))

#Permute - Returns a view of the original tensor input with its dimensions permuted.
print("Permute")
y = torch.permute(tensor,dims=(1,2,0))
print(y)
print()

#Indexing a tensor
print("Indexing")
print(tensor[1])
print(tensor[1][0])
print(tensor[1][0][1])

#Converting Tensors to numpy arrays
import numpy as np
x = torch.tensor([[1,1,1],[1,1,1]])
print(x)
num = np.array(x)
print(num)
num = num.reshape(3,2)
y = torch.tensor(num)
print(y)

print()
#Random Tensor
random_tensor = torch.rand((7,7))
print("Tensor 1 for multiplication: ")
print(random_tensor)

#Multiplication of 2 random tensors
rand_2 = torch.rand(1,7)
print("Tensor 2 for multiplication: ")
print(rand_2)
mul = random_tensor @ rand_2.T
print("After multiplication: ")
print(mul)