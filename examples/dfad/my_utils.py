import tensorlayerx as tlx
import numpy as np


def one_hot(index, num_classes=None):
    if len(index.shape) != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(tlx.reduce_max(index)) + 1

    out = tlx.zeros((index.shape[0], num_classes), dtype=tlx.int32, device=index.device)
    out_list = []

    reshaped_index = tlx.reshape(index, (-1, 1))

    reshaped_index = tlx.cast(reshaped_index, tlx.int32)

    updated = tlx.cast(tlx.ones((1,)), tlx.int32)

    for i in range(reshaped_index.shape[0]):
        out_list.append(tlx.ops.scatter_update(out[i], reshaped_index[i], updated))

    return tlx.convert_to_tensor(np.vstack(out_list))

