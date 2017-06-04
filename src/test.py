import os
import cv2
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer import Variable, optimizers, training, iterators, cuda
from chainer.training import extensions

from src import myUtil


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            dconv1 = L.Convolution2D(1, 16, 3),
            dconv2 = L.Convolution2D(16, 16, 1),
            dconv3 = L.Convolution2D(16, 32, 3),
            dconv4 = L.Convolution2D(32, 32, 1),
            uconv1 = L.Deconvolution2D(32, 32, 1),
            uconv2 = L.Deconvolution2D(32, 16, 3),
            uconv3 = L.Deconvolution2D(16, 16, 1),
            uconv4 = L.Deconvolution2D(16, 1, 3)
        )

    def __call__(self, X, t):
        h = self.forward(X)
        loss = F.mean_squared_error(h, t)
        #chainer.report({'loss': loss}, self)
        return loss

    def forward(self, X):
        h = F.relu(self.dconv1(X))
        h = F.relu(self.dconv2(h))
        h = F.relu(self.dconv3(h))
        h = F.relu(self.dconv4(h))
        h = F.relu(self.uconv1(h))
        h = F.relu(self.uconv2(h))
        h = F.relu(self.uconv3(h))
        h = F.sigmoid(self.uconv4(h))
        return h


class ImagesDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dir_path='./imgs/patches', dtype=np.float32):
        contours = myUtil.img_paths(os.path.join(dir_path, 'contour'))
        pencils = myUtil.img_paths(os.path.join(dir_path, 'pencil'))
        pairs = []
        for i in range(len(contours)):
            pairs.append((contours[i], pencils[i]))
        self._pairs = pairs[0:len(pairs)-1]
        self._dtype = dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path_c, path_p = self._pairs[i]
        img_p = cv2.imread(path_p, cv2.IMREAD_GRAYSCALE) / 255
        img_p = img_p.reshape(1, img_p.shape[0], img_p.shape[1]).astype(self._dtype)
        img_c = cv2.imread(path_c, cv2.IMREAD_GRAYSCALE)/255
        img_c = img_c.reshape(1, img_c.shape[0], img_c.shape[1]).astype(self._dtype)
        return img_p, img_c

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
    gpu_device = 0
    cuda.get_device_from_id(device_id=gpu_device).use()
    model = CNN()
    model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    train_data = ImagesDataset(dtype=cuda.cupy.float32)

    n_epoch = 1000
    batch_size = 3

    for epoch in range(1, n_epoch+1):
        print("epoch:", epoch)
        perm = np.random.permutation(len(train_data))
        sum_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = [train_data.get_example(j) for j in perm[i:i+batch_size]]
            x_batch = np.asarray([batch[j][0] for j in range(len(batch))])
            x_batch = cuda.to_gpu(x_batch)
            x_batch = Variable(x_batch)
            y_batch = np.asarray([batch[j][1] for j in range(len(batch))])
            y_batch = cuda.to_gpu(y_batch)
            y_batch = Variable(y_batch)

            optimizer.zero_grads()
            loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)
        if epoch%10 == 0:
            test_data = cuda.to_gpu(np.asarray([train_data.get_example(len(train_data) - 1)[0]]))
            test_result = cuda.to_cpu(model.forward(Variable(test_data)).data)[0][0]
            filepath = os.path.abspath("./result/" + str(epoch) + ".png")
            print(test_result.shape, test_result)
            cv2.imwrite(filepath, test_result * 255)
        print("train mean loss:", sum_loss/len(train_data))

    """
    train_iter = iterators.SerialIterator(train_data, batch_size=5, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_device)
    trainer = training.Trainer(updater, (10, 'epoch'), out="result")

    trainer.extend(extensions.Evaluator(train_iter, model, device=gpu_device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(["epoch", "main/loss"]))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    """

if __name__ == '__main__':
    #TestCNN()
    #TestDataset()
    main()
    pass
