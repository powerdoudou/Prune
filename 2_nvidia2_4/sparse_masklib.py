import sys
import torch
import numpy as np
from itertools import permutations
valid_m4n2_1d_patterns = None
#自己实现一个ASP mask
# mask
# 1. weight - >[nx4] 
# 2. pattern ->[4 x 6](2:4)
# 3. choice max_index 
# 4. feedback max_index back to mask
# compute_valid_1d_patterns & mn_1d_best
# create_mask n_1d_best(mat,4,2)
def compute_valid_1d_patterns(m,n): 
    global valid_m4n2_1d_patterns
    if m==4 and n==2 and valid_m4n2_1d_patterns is not None:
        return valid_m4n2_1d_patterns
    
    patterns=torch.zeros(m)
    patterns[:n]=1
    valid_patterns=torch.Tensor(list(set(permutations(patterns.tolist()))))
    if m == 4 and n == 2:
      valid_m4n2_1d_patterns = valid_patterns
    return valid_patterns.to('cuda')
    
def reshape_1d(matrix,m):#[15 3] 
    # If not a nice multiple of m, fill with zeroes.
    if matrix.shape[1] % m > 0:
        mat=torch.cuda.FloatTensor(
            matrix.shape[0], # 15
            matrix.shape[1] + (m - matrix.shape[1] % m) 
        ).fill_(0)
        mat[:, : matrix.shape[1]] = matrix
        shape = mat.shape
        return mat.view(-1, m), shape 
    else:
        return matrix.view(-1, m), shape

def mn_1d_best(matrix,m,n):
    patterns=compute_valid_1d_patterns(m,n)
    mat, shape = reshape_1d(matrix, m)
    
    mask = torch.cuda.IntTensor(mat.shape).fill_(1).view(-1, m)
   
    # mat,shape=reshape_1d(matrix,m)
    pmax=torch.argmax(torch.matmul(mat.abs(),patterns.t()),dim=1)
    mask[:]=patterns[pmax[:]]
    mask=mask.view(mat.shape)
    return mask
    

def m4n2_1d(mat,density):
    return mn_1d_best(mat,4,2)
def create_mask(tensor,pattern="m4n2_1d", density=0.5):
    shape = tensor.shape
    ttype = tensor.type()
    t = tensor.float().contiguous()

    # len(shape) == 2:
    t = t.view(shape[0], shape[1])
    func = getattr(sys.modules[__name__], pattern, None) # getattr() asks for the name of a thing we're looking for (like a function or an attribute in a module), and if it finds it, we can use it later in our code.
    mask = func(t, density) # func here is m4n2_1d func
    return mask.type(ttype)


if __name__=='__main__':
    weight=torch.rand(13,3).to('cuda')
    # mask = torch.cuda.IntTensor(weight.shape).fill_(1).view(-1, 4)
    matrix=create_mask(weight)
    print(matrix)


    
 