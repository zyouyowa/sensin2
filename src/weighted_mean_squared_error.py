import numpy
import cupy

from chainer import function
from chainer.utils import type_check
from chainer import cuda


class WeightedMeanSquaredError(function.Function):
    def __init__(self, alpha, beta, dh, bh):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dh = dh
        self.bh = bh

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def weight(self, x1):
        M = numpy.ones(x1.shape, x1.dtype)
        for batch in range(x1.shape[0]):
            for ch in range(x1.shape[1]):
                padded_x1 = numpy.pad(x1[batch][ch], self.dh, 'constant', constant_values=0)
                for u in range(self.dh, x1.shape[2] + self.dh):
                    for v in range(self.dh, x1.shape[3] + self.dh):
                        if padded_x1[u, v] == 1:
                            continue
                        part_x1 = padded_x1[u - self.dh: u + self.dh + 1, v - self.dh:v + self.dh + 1]
                        histogram = numpy.histogram(part_x1, self.bh)
                        for bin_i in range(1, len(histogram[1])):
                            if padded_x1[u, v] < histogram[1][bin_i]:
                                H_I_u_v = histogram[0][bin_i - 1]  # /part_x1.size
                                M_u_v = min(self.alpha * numpy.exp(-H_I_u_v) + self.beta, 1)
                                M[batch][ch][u - self.dh, v - self.dh] = M_u_v
                                break
        return M

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.M = self.weight(x1)
        self.diff = x0 - x1
        self.wdiff = self.M * self.diff
        wdiff = self.wdiff.ravel()
        return wdiff.dot(wdiff),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.M = cuda.to_gpu(self.weight(cuda.to_cpu(x1)))
        self.diff = x0 - x1
        self.wdiff = self.M * self.diff
        wdiff = self.wdiff.ravel()
        return wdiff.dot(wdiff),

    def backward(self, inputs, gy):
        gx0 = 2. * gy[0] * self.M * self.wdiff
        return gx0, -gx0


def weighted_mean_squared_error(x0, x1, alpha, beta, dh, bh):
    """Weighted Mean squared error function.

    This function computes mean weighted squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return WeightedMeanSquaredError(alpha, beta, dh, bh)(x0, x1)
