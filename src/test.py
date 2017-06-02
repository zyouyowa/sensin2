import os

import chainer
import chainer.links as L
import cv2
import numpy as np
from chainer import Chain
from chainer import Variable, optimizers

from src import myUtil


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            dconv1 = L.Convolution2D(1, 48, 3),
            dconv2 = L.Convolution2D(48, 128, 5),
            uconv1 = L.Deconvolution2D(128, 48, 5),
            uconv2 = L.Deconvolution2D(48, 1, 3)
        )

    def __call__(self, X):
        h = self.dconv1(X)
        h = self.dconv2(h)
        h = self.uconv1(h)
        h = self.uconv2(h)
        return h

class ImagesDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dir_path='./imgs/patches', dtype=np.float32):
        contours = myUtil.img_paths(os.path.join(dir_path, 'contour'))
        pencils = myUtil.img_paths(os.path.join(dir_path, 'pencil'))
        pairs = []
        for i in range(len(contours)):
            pairs.append((contours[i], pencils[i]))
        self._pairs = pairs
        self._dtype = dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path_c, path_p = self._pairs[i]
        img_c = cv2.imread(path_c, cv2.IMREAD_GRAYSCALE)/255
        img_c = img_c.reshape(1, img_c.shape[0], img_c.shape[1])
        img_p = cv2.imread(path_p, cv2.IMREAD_GRAYSCALE)/255
        img_p = img_p.reshape(1, img_p.shape[0], img_p.shape[1])
        return img_c, img_p

def TestCNN():
    img = cv2.imread(os.path.abspath("./imgs/patches/contour/0.png"), cv2.IMREAD_GRAYSCALE)
    data = np.asarray(img, np.float32).reshape((1, 1, 84, 147)) / 255
    X = Variable(data)
    model = CNN()
    y = model.forward(X)

def TestDataset():
    train = ImagesDataset()
    cv2.namedWindow('test')
    img = train.get_example(len(train)-1)[0]
    cv2.imshow('test', img.reshape(img.shape[1], img.shape[2]))
    cv2.waitKey()
    cv2.destroyAllWindows()
    #train, test = chainer.datasets.get_cifar10()
    #print(train._datasets[0][0].shape)
    #print(np.rollaxis(train._datasets[0][0], 0, 3).shape)

def main():
    model = CNN()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    for i in range(56):
        contour_path = os.path.abspath("./imgs/patches/contour/" + i + ".png")
        pencil_path = os.path.abspath("./imgs/patches/contour/" + i + ".png")
        contour_img = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        contour_img = contour_img.reshape(1, contour_img.shape[0], contour_img.shape[1])
        pencil_img = cv2.imread(pencil_path, cv2.IMREAD_GRAYSCALE)
        pencil_img = pencil_img.reshape(1, pencil_img.shape[0], pencil_img.shape[1])


if __name__ == '__main__':
    #TestCNN()
    TestDataset()
    pass
