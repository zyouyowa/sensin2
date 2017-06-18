import os
import cv2
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer import Variable, optimizers, training, iterators, cuda
from chainer.training import extensions
from src.main import ImagesDataset, SketchSimplification

from src import myUtil

def TestCNN():
    img = cv2.imread(os.path.abspath("./imgs/patches/contour/2200640_p0_2_5.jpg"), cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    data = np.asarray(img, np.float32).reshape((1, 1, 256, 256)) / 255
    X = Variable(data)
    y1 = L.Convolution2D(1, 48, 5, 2, 2)(X)
    print(y1)
    y2 = L.Convolution2D(48, 128, 3, 1, 1)(y1)
    print(y2)
    y3 = L.Deconvolution2D(128, 48, ksize=4, stride=2, pad=1)(y2)
    print(y3)
    #model = CNN()
    #y = model.forward(X)

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
    model = SketchSimplification()
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
    TestCNN()
    #TestDataset()
    #main()
    pass
