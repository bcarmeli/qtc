import torch
from torch import Tensor

from quantized import QuantizeFunction


def normalize(tensor: Tensor):
    # If all message elements are identical, return 0.5
    min_elem = tensor.min()
    # Can't change inplace because of the grads
    tensor = tensor - min_elem
    # max_elem = message.max()
    if torch.all(tensor == 0):
        tensor += 0.5
    else:
        max_elem = tensor.max()
        tensor = tensor / max_elem
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1)
    return tensor


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    batch_size = 2
    embedding_len = 5
    quantizer = QuantizeFunction.apply
    tensor1 = torch.rand((batch_size, embedding_len))
    print(f'Rand: {tensor1}')
    tensor1 = normalize(tensor1)
    print(f'Normalized: {tensor1}')
    tensor1 = quantizer(tensor1, 1.0, 0, torch.quint8)
    print(f'Quantized: {tensor1}')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
