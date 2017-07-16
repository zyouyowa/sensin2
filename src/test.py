import numpy
import cv2
import os
import chainer
from chainer import serializers, cuda
from src.main import SketchSimplification

H = 256
W = 256

def devide_img(img):
    global H, W
    h, w = img.shape
    h_size, w_size = h // H, w // W
    patches = numpy.zeros((h_size, w_size, H, W), numpy.float32)
    for i in range(h_size):
        for j in range(w_size):
            patches[i, j] = img[i*H:(i+1)*H, j*W:(j+1)*W]
    return patches, (h_size, w_size)

def main():
    global H, W
    gpu_device = 0
    cuda.get_device_from_id(gpu_device)
    model = SketchSimplification(6, -2, 2, 10)
    serializers.load_npz('./model_mse.npz', model)
    model.to_gpu(gpu_device)

    path = os.path.abspath("./imgs/test_in3.png")
    in_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
    out_img = numpy.copy(in_img)

    in_patches, size = devide_img(in_img)
    for i in range(size[0]):
        for j in range(size[1]):
            in_name = 'in_' + str(i) + '_' + str(j)
            cv2.namedWindow(in_name)
            cv2.imshow(in_name, in_patches[i, j])

            in_patch = in_patches[i, j]
            in_patch = in_patch.reshape(1, 1, in_patch.shape[0], in_patch.shape[1])
            x = chainer.Variable(cuda.to_gpu(in_patch))
            y = cuda.to_cpu(model.forward(x).data)[0][0] * 255

            out_name = 'out_' + str(i) + '_' + str(j)
            cv2.namedWindow(out_name)
            cv2.imshow(out_name, y)

            out_img[i*H:(i+1)*H, j*W:(j+1)*W] = y

    cv2.namedWindow('in')
    cv2.imshow('in', in_img)
    cv2.namedWindow('out')
    cv2.imshow('out', out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
