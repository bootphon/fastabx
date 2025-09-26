import torch

from fastabx.dtw import dtw

torch.manual_seed(0)
x = torch.randn((3, 4))
print(x)
print(dtw(x))
if torch.cuda.is_available():
    print(dtw(x.cuda()))
