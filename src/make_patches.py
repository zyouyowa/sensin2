import cv2
import os
from src import myUtil

H = 256
W = 256


def check_minimum(path):
    global H, W
    files = os.listdir(path)
    min_h, min_w = 9999, 9999
    for f in files:
        file = os.path.abspath(os.path.join(path, f))
        if not myUtil.is_img(file):
            continue
        img = cv2.imread(file)
        h, w, ch = img.shape
        if h < min_h:
            min_h = h
        if w < min_w:
            min_w = w
    H = min_h
    W = min_w
    print("H=" + str(H), "W=" + str(W))

def make_patches(path, path_patches):
    global H, W
    files = os.listdir(path)
    for f in files:
        if not myUtil.is_img(f):
            continue
        file = os.path.abspath(os.path.join(path, f))
        print(file)
        img = cv2.imread(file)
        h, w, ch = img.shape
        h_size, w_size = h // H, w // W
        h_garbage, w_garbage = (h % H) // 2, (w % W) // 2
        print(h, w, h_size, w_size, h_garbage, w_garbage)
        for i in range(h_size):
            i += 1
            for j in range(w_size):
                j += 1
                # _garbage分は作成されるパッチには使用されない
                h_begin = h_garbage + (i-1) * H
                h_end = h_garbage + i * H
                w_begin = w_garbage + (j-1) * W
                w_end = w_garbage + j * W
                patch = img[h_begin:h_end, w_begin:w_end]
                name, ext = os.path.splitext(f)
                patch_name = name + '_' + str(i) + '_' + str(j) + ext
                patch_path = os.path.abspath(os.path.join(path_patches, patch_name))
                print("to: " + patch_path)
                cv2.imwrite(patch_path, patch)


if __name__ == '__main__':
    #check_minimum("./imgs/pencil")
    make_patches("./imgs/contour", "./imgs/patches/contour")
    make_patches("./imgs/pencil", "./imgs/patches/pencil")
