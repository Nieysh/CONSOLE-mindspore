import random
import numpy as np
import mindspore
import mindspore.ops as ops

def set_random_seed(seed):
    mindspore.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    A = ops.arange(size, dtype=mindspore.int64).unsqueeze(0).tile((batch_size, 1))
    B = mindspore.Tensor(np.array(length) - 1, mindspore.int64).unsqueeze(1)
    mask = ( A > B )
    tmp_output = mask

    return tmp_output
