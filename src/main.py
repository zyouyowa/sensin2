import os
import cv2
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable, optimizers, cuda, serializers

from src import myUtil


class SketchSimplification(Chain):
    def __init__(self):
        ch1 = 48
        ch2 = 128
        ch3 = 256
        ch4 = 512
        ch5 = 1024
        ch6 = 24
        super(SketchSimplification, self).__init__(
            dconv1      = L.Convolution2D(1, ch1, ksize=5, stride=2, pad=2),
            fconv_d1_1  = L.Convolution2D(ch1, ch2, ksize=3, stride=1, pad=1),
            fconv_d1_2  = L.Convolution2D(ch2, ch2, ksize=3, stride=1, pad=1),

            dconv2      = L.Convolution2D(ch2, ch3, ksize=3, stride=2, pad=1),
            fconv_d2_1  = L.Convolution2D(ch3, ch3, ksize=3, stride=1, pad=1),
            fconv_d2_2  = L.Convolution2D(ch3, ch3, ksize=3, stride=1, pad=1),

            dconv3      = L.Convolution2D(ch3, ch3, ksize=3, stride=2, pad=1),
            fconv_d3_1  = L.Convolution2D(ch3, ch4, ksize=3, stride=1, pad=1),
            fconv_d3_2  = L.Convolution2D(ch4, ch5, ksize=3, stride=1, pad=1),
            fconv_d3_3  = L.Convolution2D(ch5, ch5, ksize=3, stride=1, pad=1),
            fconv_d3_4  = L.Convolution2D(ch5, ch5, ksize=3, stride=1, pad=1),
            fconv_d3_5  = L.Convolution2D(ch5, ch5, ksize=3, stride=1, pad=1),
            fconv_d3_6  = L.Convolution2D(ch5, ch4, ksize=3, stride=1, pad=1),
            fconv_d3_7  = L.Convolution2D(ch4, ch3, ksize=3, stride=1, pad=1),

            uconv1      = L.Deconvolution2D(ch3, ch3, ksize=4, stride=2, pad=1),
            fconv_u1_1  = L.Convolution2D(ch3, ch3, ksize=3, stride=1, pad=1),
            fconv_u1_2  = L.Convolution2D(ch3, ch2, ksize=3, stride=1, pad=1),

            uconv2      = L.Deconvolution2D(ch2, ch2, ksize=4, stride=2, pad=1),
            fconv_u2_1  = L.Convolution2D(ch2, ch2, ksize=3, stride=1, pad=1),
            fconv_u2_2  = L.Convolution2D(ch2, ch1, ksize=3, stride=1, pad=1),

            uconv3      = L.Deconvolution2D(ch1, ch1, ksize=4, stride=2, pad=1),
            fconv_u3_1  = L.Convolution2D(ch1, ch6, ksize=3, stride=1, pad=1),
            fconv_u3_2  = L.Convolution2D(ch6, 1, ksize=3, stride=1, pad=1),

            bnormd1     = L.BatchNormalization(ch1),
            bnormd1_1   = L.BatchNormalization(ch2),
            bnormd1_2   = L.BatchNormalization(ch2),

            bnormd2     = L.BatchNormalization(ch3),
            bnormd2_1   = L.BatchNormalization(ch3),
            bnormd2_2   = L.BatchNormalization(ch3),

            bnormd3     = L.BatchNormalization(ch3),
            bnormd3_1   = L.BatchNormalization(ch4),
            bnormd3_2   = L.BatchNormalization(ch5),
            bnormd3_3   = L.BatchNormalization(ch5),
            bnormd3_4   = L.BatchNormalization(ch5),
            bnormd3_5   = L.BatchNormalization(ch5),
            bnormd3_6   = L.BatchNormalization(ch4),
            bnormd3_7   = L.BatchNormalization(ch3),

            bnormu1     = L.BatchNormalization(ch3),
            bnormu1_1   = L.BatchNormalization(ch3),
            bnormu1_2   = L.BatchNormalization(ch2),

            bnormu2     = L.BatchNormalization(ch2),
            bnormu2_1   = L.BatchNormalization(ch2),
            bnormu2_2   = L.BatchNormalization(ch1),

            bnormu3     = L.BatchNormalization(ch1),
            bnormu3_1   = L.BatchNormalization(ch6)
        )

    def __call__(self, X, t):
        h = self.forward(X)
        loss = F.mean_squared_error(h, t)
        #chainer.report({'loss': loss}, self)
        return loss

    def forward(self, X):
        h = F.relu(self.bnormd1(self.dconv1(X)))
        h = F.relu(self.bnormd1_1(self.fconv_d1_1(h)))
        h = F.relu(self.bnormd1_2(self.fconv_d1_2(h)))
        h = F.relu(self.bnormd2(self.dconv2(h)))
        h = F.relu(self.bnormd2_1(self.fconv_d2_1(h)))
        h = F.relu(self.bnormd2_2(self.fconv_d2_2(h)))
        h = F.relu(self.bnormd3(self.dconv3(h)))
        h = F.relu(self.bnormd3_1(self.fconv_d3_1(h)))
        h = F.relu(self.bnormd3_2(self.fconv_d3_2(h)))
        h = F.relu(self.bnormd3_3(self.fconv_d3_3(h)))
        h = F.relu(self.bnormd3_4(self.fconv_d3_4(h)))
        h = F.relu(self.bnormd3_5(self.fconv_d3_5(h)))
        h = F.relu(self.bnormd3_6(self.fconv_d3_6(h)))
        h = F.relu(self.bnormd3_7(self.fconv_d3_7(h)))
        h = F.relu(self.bnormu1(self.uconv1(h)))
        h = F.relu(self.bnormu1_1(self.fconv_u1_1(h)))
        h = F.relu(self.bnormu1_2(self.fconv_u1_2(h)))
        h = F.relu(self.bnormu2(self.uconv2(h)))
        h = F.relu(self.bnormu2_1(self.fconv_u2_1(h)))
        h = F.relu(self.bnormu2_2(self.fconv_u2_2(h)))
        h = F.relu(self.bnormu3(self.uconv3(h)))
        h = F.relu(self.bnormu3_1(self.fconv_u3_1(h)))
        h = F.sigmoid(self.fconv_u3_2(h))
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
        img_p = cv2.imread(path_p, cv2.IMREAD_GRAYSCALE) / 255
        img_p = img_p.reshape(1, img_p.shape[0], img_p.shape[1]).astype(self._dtype)
        img_c = cv2.imread(path_c, cv2.IMREAD_GRAYSCALE)/255
        img_c = img_c.reshape(1, img_c.shape[0], img_c.shape[1]).astype(self._dtype)
        return img_p, img_c

def main():
    gpu_device = 0
    cuda.get_device_from_id(device_id=gpu_device).use()
    model = SketchSimplification()
    model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    train_data = ImagesDataset(dir_path='./imgs/trains', dtype=cuda.cupy.float32)
    test_data = ImagesDataset(dir_path='./imgs/tests', dtype=cuda.cupy.float32)

    n_epoch = 30
    batch_size = 3

    for epoch in range(1, n_epoch+1):
        print('epoch:', epoch)
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
        if epoch%1 == 0:
            test_img = cuda.to_gpu(np.asarray([test_data.get_example(len(test_data) - 1)[0]]))
            test_result = cuda.to_cpu(model.forward(Variable(test_img)).data)[0][0]
            filepath = os.path.abspath('./result/' + str(epoch) + '.png')
            print(test_result.shape, test_result)
            cv2.imwrite(filepath, test_result * 255)
        print('train mean loss:', sum_loss/len(train_data))

    model.to_cpu()
    serializers.save_npz('model1.npz', model)

if __name__ == '__main__':
    main()
    pass
