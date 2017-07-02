import numpy
import cv2
import os

def weight(x1, alpha, beta, dh, bh):
    padded_x1 = numpy.pad(x1, dh, 'constant', constant_values=0)
    M = numpy.ones(x1.shape, x1.dtype)
    for u in range(dh, x1.shape[0] + dh):
        for v in range(dh, x1.shape[1] + dh):
            if padded_x1[u, v] == 1:
                continue
            part_x1 = padded_x1[u-dh : u+dh+1, v-dh :v+dh+1]
            histogram = numpy.histogram(part_x1, bh)
            for bin_i in range(1, len(histogram[1])):
                if padded_x1[u, v] < histogram[1][bin_i]:
                    H_I_u_v = histogram[0][bin_i - 1]#/part_x1.size
                    M_u_v = min(alpha * numpy.exp(-H_I_u_v) + beta, 1)
                    M[u - dh, v - dh] = M_u_v
                    break
    return M
    #self.diff = x0 - x1
    #w_diff = (M * self.diff).ravel()
    #return numpy.array(w_diff.dot(w_diff) / w_diff.size, dtype=w_diff.dtype),

def main():
    #path = os.path.abspath("./imgs/trains/contour/19.jpg")
    path = os.path.abspath("./imgs/ayano.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
    def downer(elem, threthold):
        if elem < threthold:
            return 0
        else:
            return elem
    vdowner = numpy.vectorize(downer)
    img_2 = vdowner(img, 0.9)
    m = weight(img_2, 6, -2, 2, 10)
    cv2.namedWindow('test2')
    cv2.namedWindow('weight')
    cv2.imshow('test2', img_2)
    cv2.imshow('weight', m)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    main()
