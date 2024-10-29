


import torch
nid = torch.tensor([1,2,6,5,2,6,7,9,6,9])

def uni_inv(tensor):
    uni, inv = torch.unique(tensor, return_inverse=True)
    uni, inv = uni.cpu(), inv.cpu()
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)

    return uni,inv,perm

