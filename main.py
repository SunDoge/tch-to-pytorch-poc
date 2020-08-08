import tch
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

dlt = tch.eye(10)

x = from_dlpack(dlt)

print(x)

y = torch.rand(2,3)

tch.print(to_dlpack(y))