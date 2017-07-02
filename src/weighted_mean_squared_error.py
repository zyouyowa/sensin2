import numpy

from chainer import function
from chainer.utils import type_check


class WeightedMeanSquaredError(function.Function):
    def __init__(self, alpha, beta, dh, bh):
        super.__init__()
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
        padded_x1 = numpy.pad(x1, self.dh, 'constant', constant_values=0)
        self.M = numpy.ones(x1.shape, x1.dtype)
        for u in range(self.dh, x1.shape[0] + self.dh):
            for v in range(self.dh, x1.shape[1] + self.dh):
                if padded_x1[u, v] == 1:
                    continue
                part_x1 = padded_x1[u - self.dh:u + self.dh + 1, v - self.dh:v + self.dh + 1]
                histogram = numpy.histogram(part_x1, self.bh)
                for bin_i in range(1, len(histogram[1])):
                    if padded_x1[u, v] < histogram[1][bin_i]:
                        H_I_u_v = histogram[0][bin_i-1]
                        M_u_v = min(self.alpha * numpy.exp(-H_I_u_v) + self.beta, 1)
                        self.M[u-self.dh, v-self.dh] = M_u_v
                        break
        self.diff = x0 - x1
        return numpy.linalg.norm(self.M * self.diff)

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
