import torch
#Debugging
def Duplicate_checking(input):
    num_ts = 1
    rep = []
    for idx, i in enumerate(input.shape):
        num_ts *= (i - 1) if idx == 0 else i
        rep.append(1)
    rep[0] *= input.shape[0] - 1
    sample = input[0].repeat(rep)
    overlap = input[1:] == sample
    return overlap.sum() / num_ts
