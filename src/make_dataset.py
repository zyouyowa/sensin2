import os
import cv2
from src.contour import make_contour_image
from src.pencil import make_pencil_sketch_image
from src import myUtil


def make_dataset(path_color,  path_contour, path_pencil):
    src_files = os.listdir(path_color)
    for s in src_files:
        color_src_path = os.path.abspath(os.path.join(path_color, s))
        if not myUtil.is_img(color_src_path):
            continue
        contour_img = make_contour_image(color_src_path)
        contour_src_path = os.path.abspath(os.path.join(path_contour, s))
        cv2.imwrite(contour_src_path, contour_img)
        pencil_img = make_pencil_sketch_image(contour_src_path)
        cv2.imwrite(os.path.abspath(os.path.join(path_pencil, s)), pencil_img)

if __name__ == '__main__':
    make_dataset("./imgs/colors", "./imgs/contour", "./imgs/pencil")
