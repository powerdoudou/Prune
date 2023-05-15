import torch


weight=torch.tensor([[10,3,30,9,6],[1,34,35,8,2],[1,8,30,90,6]])
mask=torch.ones_like(weight, dtype=bool)
indices=torch.tensor([0,6,7,11])
mask.view(-1)[indices] = False
weight1=mask*weight+torch.rand(3,5)
# print("mask:....")
# print(mask)
# print("weight:....")
# print(weight1)
mutation_rate=0.3
num_true=torch.count_nonzero(mask)
mutate_num=int(mutation_rate*num_true)

true_indices_2d=torch.where(mask==True)
print(true_indices_2d)
true_element_1d_idx_prune=torch.topk(weight[true_indices_2d],mutate_num,largest=False)[1]

print(true_element_1d_idx_prune)

for i in true_element_1d_idx_prune:
    mask[true_indices_2d[0][i], true_indices_2d[1][i]] = False
print("mask:....")
print(mask)
#regrowing
false_indices = torch.nonzero(~mask)
print(false_indices)
random_indices = torch.randperm(false_indices.shape[0])[:mutate_num]
print("random_indices:................................")
print(random_indices)
regrow_indices = false_indices[random_indices]
print("regrow_indices:................................")
print(regrow_indices)
for regrow_idx in regrow_indices:
    mask[tuple(regrow_idx)] = True
print("mask:................................")
print(mask)
