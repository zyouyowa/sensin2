import numpy

from chainer import function
from chainer.utils import type_check


class WeightedMeanSquaredError(function.Function):
    def __init__(self, bins, alpha, beta, dh, bh):
        super.__init__()
        self.bins = bins
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

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        histogram = numpy.histogram(x1, self.bins)
        # TODO: H(I, u, v)の計算
        # TODO: Mの計算
        diff = self.diff.ravel()
        return numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        # TODO: ヒストグラムの計算
        # TODO: H(I, u, v)の計算
        # TODO: Mの計算
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diff
        return gx0, -gx0


def weighted_mean_squared_error(x0, x1,  bins, alpha, beta, dh, bh):
    """Weighted Mean squared error function.

    This function computes mean weighted squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return WeightedMeanSquaredError()(x0, x1, bins, alpha, beta, dh, bh)
