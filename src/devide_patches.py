import os
import numpy as np
from src.myUtil import img_paths, is_img

def shuffle(train_n, test_n):
    if train_n < 1 or test_n < 1:
        print('Error! Too small datasets!')

    c_src_path = './imgs/patches/contour'
    contour_srcs = img_paths(c_src_path)
    if train_n + test_n > len(contour_srcs):
        print('Error! train_n and test_n are too large!')
    p_src_path = './imgs/patches/pencil'
    pencil_srcs = img_paths(p_src_path)

    need_dirs = [
        './imgs/trains',
        './imgs/trains/contour',
        './imgs/trains/pencil',
        './imgs/tests',
        './imgs/tests/contour',
        './imgs/tests/pencil',
    ]
    for need_dir in need_dirs:
        if not os.path.exists(need_dir):
            os.mkdir(need_dir)
            os.chmod(need_dir, 755)
        for f in os.listdir(need_dir):
            if not is_img(f):
                continue
            os.remove(os.path.join(need_dir, f))

    perm = np.random.permutation(len(contour_srcs))
    for i in range(train_n):
        c_path = contour_srcs[perm[i]]
        p_path = pencil_srcs[perm[i]]
        _, ext = os.path.splitext(c_path)
        os.symlink(c_path, os.path.join(need_dirs[1], str(i) + ext))
        os.symlink(p_path, os.path.join(need_dirs[2], str(i) + ext))
    for i in range(train_n, train_n + test_n):
        c_path = contour_srcs[perm[i]]
        p_path = pencil_srcs[perm[i]]
        _, ext = os.path.splitext(c_path)
        os.symlink(c_path, os.path.join(need_dirs[4], str(i - train_n) + ext))
        os.symlink(p_path, os.path.join(need_dirs[5], str(i - train_n) + ext))


if __name__ == '__main__':
    shuffle(128, 16)
