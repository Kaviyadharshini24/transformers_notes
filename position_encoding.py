import torch 
import torch.nn as nn
max_seequence_len=23
d_model=12
even_i=torch.arange(0, d_model, 2).float()
# print(even_i)
even_denominator=torch.pow(10000,even_i/d_model)
# print(even_denominator)
odd_i=torch.arange(1,d_model,2).float()
# print(odd_i)
even_dominator=torch.pow(10000,odd_i-1/d_model)
# print(even_dominator)
denominator=even_denominator
position=torch.arange(0,max_seequence_len,dtype=torch.float).reshape(max_seequence_len,1)
# print(position)
even_pe=torch.sin(position/denominator)
odd_pe=torch.cos(position/denominator)
# print(even_pe)
# print(odd_pe)
stacks=torch.stack([even_pe,odd_pe],dim=2)
# print(stacks)
pe=torch.flatten(stacks,start_dim=1,end_dim=2)
print(pe)


