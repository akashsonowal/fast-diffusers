from typing import Sequence
import torch 

TensorLikeType = torch.Tensor


def apply_memory_format(m, memory_format=torch.preserve_format):

    def convert(t):
        pass 

    return m._apply(convert)