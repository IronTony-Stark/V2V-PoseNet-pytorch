import numpy as np
import torch
import torch.nn as nn


def softargmax2d(input, beta=100):
    *_, h, w = input.shape

    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result


def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def gaussian_2d(shape, center, sigma=1.0):
    xs, ys = np.meshgrid(
        np.arange(0, shape[1], step=1.0, dtype=np.float32),
        np.arange(0, shape[0], step=1.0, dtype=np.float32))

    alpha = -0.5 / (sigma ** 2)
    res = np.exp(alpha * ((xs - center[1]) ** 2 + (ys - center[0]) ** 2))
    return res


# Testing
def test_softargmax1d():
    x = torch.tensor([1.5, 2.5, 3.5, 100.5, 3.5, 2.5, 1.5], dtype=torch.float32)
    argmax = softargmax1d(x)

    assert argmax == 3


def test_softargmax2d():
    x = torch.tensor(gaussian_2d(shape=(20, 30), center=(4, 13)))
    argmax = softargmax2d(x).numpy()
    assert np.all(argmax[0, :] == [4, 13])



class A:
    def __init__(self, a):
        self.a = a

class B(A):
    def __init__(self, a, b):
        super().__init__(a)
        self.a = b


if __name__ == "__main__":
    b = B(1, 2)
    print(b.a)
    # test_softargmax1d()
    # test_softargmax2d()
    # print("All tests passed")
