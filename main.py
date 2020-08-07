import tch
import torch
from torch.utils.dlpack import from_dlpack

dlt = tch.eye(10)

x = from_dlpack(dlt)

print(x)