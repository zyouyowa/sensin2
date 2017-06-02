import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class DownUpConvModel(Chain):
    def __init__(self):
        super(DownUpConvModel, self).__init__(
            dconv1 = L.Convolution2D(3, 48, 5, 2),
            fconv1 = L.Convolution2D(48, 48, 3, 1),
            uconv1 = L.Deconvolution2D(48, 3, 4, 2)
        )

    def __call__(self, X_data, y_data):
        h = F.relu(self.dconv1(X_data))
        h = F.relu(self.fconv1(h))
        h = F.sigmoid(self.uconv1(h))
        self.loss = F.mean_squared_error(h, y_data)
        return self.loss

def make_dataset():
    pass

def main():
    model = DownUpConvModel()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    batch_size = 6
    for epoch in range(5):
        print("epoch: %d", epoch)
        indexes = np.random.permutation()




if __name__ == '__main__':
    main()

